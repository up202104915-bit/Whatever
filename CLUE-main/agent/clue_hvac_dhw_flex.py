"""CLUE extension utilities for HVAC + DHW flexibility envelope generation.

This module is designed to sit next to the original CLUE code and provide:
1) Building model inspection (epJSON) for HVAC and DHW control points.
2) Comfort-prioritized flexibility envelope computation.
3) Export helpers for downstream optimization tools such as GAMS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComfortConfig:
    """Configuration for comfort-aware flexibility calculations."""

    # Occupied thermal comfort band [degC]
    hvac_occ_band: Tuple[float, float] = (20.0, 24.0)
    # Unoccupied thermal comfort band [degC]
    hvac_unocc_band: Tuple[float, float] = (16.0, 30.0)
    # DHW tank comfort/safety band [degC]
    dhw_band: Tuple[float, float] = (50.0, 60.0)
    # Optional scaling; >1 gives more flexibility when unoccupied
    unoccupied_flex_multiplier: float = 1.30
    # kW flexibility per degree of thermal slack (HVAC)
    hvac_sensitivity_kw_per_c: float = 2.0
    # kW flexibility per degree of thermal slack (DHW)
    dhw_sensitivity_kw_per_c: float = 1.2
    # Hard limits to avoid unrealistic envelopes
    max_hvac_shift_kw: float = 40.0
    max_dhw_shift_kw: float = 20.0
    # Occupancy threshold for "occupied"
    occupied_threshold: float = 0.01


def inspect_epjson_hvac_dhw(model_path: str | Path) -> Dict[str, object]:
    """Inspect an EnergyPlus epJSON model and extract HVAC + DHW hooks.

    Args:
        model_path: Path to an epJSON file.

    Returns:
        Dictionary with relevant schedules and object names for control.
    """
    path = Path(model_path)
    if path.suffix.lower() not in {".epjson", ".json"}:
        raise ValueError(
            f"Expected an epJSON file, got: {path}. "
            "Convert IDF to epJSON first for this utility."
        )

    with path.open("r", encoding="utf-8") as f:
        building = json.load(f)

    dual_setpoints = building.get("ThermostatSetpoint:DualSetpoint", {})
    heat_sp_schedules = sorted(
        {
            obj.get("heating_setpoint_temperature_schedule_name")
            for obj in dual_setpoints.values()
            if obj.get("heating_setpoint_temperature_schedule_name")
        }
    )
    cool_sp_schedules = sorted(
        {
            obj.get("cooling_setpoint_temperature_schedule_name")
            for obj in dual_setpoints.values()
            if obj.get("cooling_setpoint_temperature_schedule_name")
        }
    )

    water_heaters = building.get("WaterHeater:Mixed", {})
    dhw_setpoint_schedules = sorted(
        {
            obj.get("setpoint_temperature_schedule_name")
            for obj in water_heaters.values()
            if obj.get("setpoint_temperature_schedule_name")
        }
    )

    water_heater_names = sorted(list(water_heaters.keys()))

    return {
        "model_path": str(path),
        "has_hvac_dual_setpoint": len(dual_setpoints) > 0,
        "has_dhw": len(water_heaters) > 0,
        "hvac_heating_schedules": heat_sp_schedules,
        "hvac_cooling_schedules": cool_sp_schedules,
        "dhw_setpoint_schedules": dhw_setpoint_schedules,
        "water_heater_names": water_heater_names,
    }


def build_action_definition_from_inspection(
    inspection: Dict[str, object],
    heating_action: str = "Office_Heating_RL",
    cooling_action: str = "Office_Cooling_RL",
    dhw_action: str = "DHW_Setpoint_RL",
) -> Dict[str, Dict[str, object]]:
    """Create Sinergym action_definition for HVAC + DHW.

    This adapts CLUE control from HVAC-only to HVAC+DHW by exposing a third
    action variable (DHW setpoint schedule).
    """

    heating_schedules = inspection.get("hvac_heating_schedules", [])
    cooling_schedules = inspection.get("hvac_cooling_schedules", [])
    dhw_schedules = inspection.get("dhw_setpoint_schedules", [])

    if not heating_schedules or not cooling_schedules:
        raise ValueError("No HVAC setpoint schedules found in building inspection.")
    if not dhw_schedules:
        raise ValueError("No DHW setpoint schedules found in building inspection.")

    return {
        str(heating_schedules[0]): {"name": heating_action, "initial_value": 21.0},
        str(cooling_schedules[0]): {"name": cooling_action, "initial_value": 25.0},
        str(dhw_schedules[0]): {"name": dhw_action, "initial_value": 60.0},
    }


def prepare_hvac_dhw_timeseries(
    monitor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize monitor output columns to a canonical HVAC+DHW frame.

    Expected source monitor columns are inferred by robust name matching.
    Output columns are:
        - zone_temp_c
        - occupancy
        - hvac_power_kw
        - dhw_tank_temp_c
        - dhw_power_kw
    """

    def _first_match(candidates: Iterable[str], needle: str) -> Optional[str]:
        for col in candidates:
            if needle in col:
                return col
        return None

    cols = list(monitor_df.columns)

    zone_temp_col = _first_match(cols, "Zone Air Temperature(")
    if zone_temp_col is None:
        raise ValueError("No 'Zone Air Temperature(...)' column found in monitor data.")

    occupancy_cols = [c for c in cols if "Zone People Occupant Count(" in c]
    hvac_power_col = _first_match(
        cols, "Facility Total HVAC Electricity Demand Rate(Whole Building)"
    )
    if hvac_power_col is None:
        raise ValueError(
            "No HVAC power column found: "
            "'Facility Total HVAC Electricity Demand Rate(Whole Building)'."
        )

    dhw_tank_col = _first_match(cols, "Water Heater Tank Temperature(")
    dhw_power_col = _first_match(cols, "Water Heater Heating Rate(")

    canonical = pd.DataFrame(index=monitor_df.index)
    canonical["zone_temp_c"] = monitor_df[zone_temp_col].astype(float)
    if occupancy_cols:
        canonical["occupancy"] = monitor_df[occupancy_cols].astype(float).sum(axis=1)
    else:
        canonical["occupancy"] = 0.0
    canonical["hvac_power_kw"] = monitor_df[hvac_power_col].astype(float) / 1000.0

    if dhw_tank_col is None:
        canonical["dhw_tank_temp_c"] = np.nan
    else:
        canonical["dhw_tank_temp_c"] = monitor_df[dhw_tank_col].astype(float)

    if dhw_power_col is None:
        canonical["dhw_power_kw"] = 0.0
    else:
        canonical["dhw_power_kw"] = monitor_df[dhw_power_col].astype(float) / 1000.0

    return canonical


def compute_comfort_priority_flex_envelope(
    timeseries: pd.DataFrame,
    comfort: ComfortConfig = ComfortConfig(),
) -> pd.DataFrame:
    """Compute a flexibility power envelope with comfort as primary constraint.

    The envelope is represented as:
        p_min_kw <= p_base_kw <= p_max_kw
    where p_min_kw and p_max_kw capture feasible downward/upward flexibility.
    """

    required_cols = {"zone_temp_c", "occupancy", "hvac_power_kw", "dhw_power_kw"}
    missing = required_cols - set(timeseries.columns)
    if missing:
        raise ValueError(f"Missing required timeseries columns: {sorted(missing)}")

    n = len(timeseries)
    if n == 0:
        raise ValueError("Timeseries is empty.")

    zone_temp = timeseries["zone_temp_c"].to_numpy(dtype=float)
    occupancy = timeseries["occupancy"].to_numpy(dtype=float)
    hvac_power_kw = np.clip(timeseries["hvac_power_kw"].to_numpy(dtype=float), 0.0, None)
    dhw_power_kw = np.clip(timeseries["dhw_power_kw"].to_numpy(dtype=float), 0.0, None)

    has_dhw_temp = "dhw_tank_temp_c" in timeseries.columns and not timeseries[
        "dhw_tank_temp_c"
    ].isna().all()
    if has_dhw_temp:
        dhw_temp = timeseries["dhw_tank_temp_c"].to_numpy(dtype=float)
    else:
        dhw_temp = np.full(n, np.nan, dtype=float)

    occupied = occupancy > comfort.occupied_threshold
    temp_low = np.where(occupied, comfort.hvac_occ_band[0], comfort.hvac_unocc_band[0])
    temp_high = np.where(occupied, comfort.hvac_occ_band[1], comfort.hvac_unocc_band[1])

    # Thermal slack from comfort limits; max-flexibility near comfort center.
    slack_to_low = np.clip(zone_temp - temp_low, 0.0, None)
    slack_to_high = np.clip(temp_high - zone_temp, 0.0, None)
    hvac_comfort_slack = np.minimum(slack_to_low, slack_to_high)

    occ_multiplier = np.where(occupied, 1.0, comfort.unoccupied_flex_multiplier)

    hvac_flex_up = np.minimum(
        comfort.max_hvac_shift_kw,
        comfort.hvac_sensitivity_kw_per_c * hvac_comfort_slack * occ_multiplier,
    )
    hvac_flex_down = np.minimum(
        comfort.max_hvac_shift_kw,
        comfort.hvac_sensitivity_kw_per_c * hvac_comfort_slack * occ_multiplier,
    )

    if has_dhw_temp:
        dhw_low, dhw_high = comfort.dhw_band
        dhw_slack_down = np.clip(dhw_temp - dhw_low, 0.0, None)
        dhw_slack_up = np.clip(dhw_high - dhw_temp, 0.0, None)

        dhw_flex_down = np.minimum(
            comfort.max_dhw_shift_kw, comfort.dhw_sensitivity_kw_per_c * dhw_slack_down
        )
        dhw_flex_up = np.minimum(
            comfort.max_dhw_shift_kw, comfort.dhw_sensitivity_kw_per_c * dhw_slack_up
        )
        dhw_violation = np.where(
            np.isnan(dhw_temp),
            0.0,
            np.maximum(dhw_low - dhw_temp, 0.0) + np.maximum(dhw_temp - dhw_high, 0.0),
        )
    else:
        dhw_flex_down = np.zeros(n, dtype=float)
        dhw_flex_up = np.zeros(n, dtype=float)
        dhw_violation = np.zeros(n, dtype=float)

    comfort_violation_hvac = np.maximum(temp_low - zone_temp, 0.0) + np.maximum(
        zone_temp - temp_high, 0.0
    )

    p_base_kw = hvac_power_kw + dhw_power_kw
    flex_down_kw = hvac_flex_down + dhw_flex_down
    flex_up_kw = hvac_flex_up + dhw_flex_up

    p_min_kw = np.maximum(0.0, p_base_kw - flex_down_kw)
    p_max_kw = p_base_kw + flex_up_kw

    out = pd.DataFrame(
        {
            "t": np.arange(1, n + 1, dtype=int),
            "p_base_kw": p_base_kw,
            "p_min_kw": p_min_kw,
            "p_max_kw": p_max_kw,
            "flex_down_kw": flex_down_kw,
            "flex_up_kw": flex_up_kw,
            "hvac_power_kw": hvac_power_kw,
            "dhw_power_kw": dhw_power_kw,
            "comfort_violation_hvac_c": comfort_violation_hvac,
            "comfort_violation_dhw_c": dhw_violation,
        }
    )
    return out


def envelope_to_gams_inc(
    envelope_df: pd.DataFrame,
    output_path: str | Path,
    set_name: str = "t",
    base_param: str = "P_BASE",
    min_param: str = "P_MIN",
    max_param: str = "P_MAX",
) -> None:
    """Export envelope to a GAMS include file (.inc)."""
    required_cols = {"t", "p_base_kw", "p_min_kw", "p_max_kw"}
    missing = required_cols - set(envelope_df.columns)
    if missing:
        raise ValueError(f"Envelope is missing required columns: {sorted(missing)}")

    lines: List[str] = []
    t_max = int(envelope_df["t"].max())
    lines.append(f"Set {set_name} /t1*t{t_max}/;")
    lines.append(f"Parameter {base_param}({set_name}), {min_param}({set_name}), {max_param}({set_name});")
    lines.append("")

    for row in envelope_df.itertuples(index=False):
        t_tag = f"t{int(row.t)}"
        lines.append(f"{base_param}('{t_tag}') = {float(row.p_base_kw):.6f};")
        lines.append(f"{min_param}('{t_tag}') = {float(row.p_min_kw):.6f};")
        lines.append(f"{max_param}('{t_tag}') = {float(row.p_max_kw):.6f};")

    out_path = Path(output_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
