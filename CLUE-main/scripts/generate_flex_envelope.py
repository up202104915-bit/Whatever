"""Generate a comfort-priority HVAC+DHW flexibility envelope for GAMS.

Usage example:
python3 scripts/generate_flex_envelope.py \
  --building sinergym/data/buildings/ASHRAE901_OfficeMedium_STD2019_Denver.epJSON \
  --monitor results/office_monitor.csv \
  --out-csv results/flex_envelope.csv \
  --out-gams results/flex_envelope.inc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agent.clue_hvac_dhw_flex import (
    ComfortConfig,
    compute_comfort_priority_flex_envelope,
    envelope_to_gams_inc,
    inspect_building_hvac_dhw,
    prepare_hvac_dhw_timeseries,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create comfort-priority flexibility power envelope."
    )
    parser.add_argument(
        "--building",
        required=True,
        help="Path to building model (.idf or .epJSON).",
    )
    parser.add_argument(
        "--monitor",
        required=True,
        help="Path to monitor CSV generated from simulation.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for the flexibility envelope.",
    )
    parser.add_argument(
        "--out-gams",
        required=True,
        help="Output .inc path for GAMS.",
    )
    parser.add_argument(
        "--occ-min",
        type=float,
        default=20.0,
        help="Occupied comfort lower bound (degC).",
    )
    parser.add_argument(
        "--occ-max",
        type=float,
        default=24.0,
        help="Occupied comfort upper bound (degC).",
    )
    parser.add_argument(
        "--dhw-min",
        type=float,
        default=50.0,
        help="DHW tank minimum comfort/safety (degC).",
    )
    parser.add_argument(
        "--dhw-max",
        type=float,
        default=60.0,
        help="DHW tank maximum comfort/safety (degC).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    inspection = inspect_building_hvac_dhw(args.building)
    print("=== Building inspection summary ===")
    print(f"Model: {inspection['model_path']}")
    print(f"Model format: {inspection['model_format']}")
    print(f"DHW detected: {inspection['has_dhw']}")
    print(f"Water heaters: {inspection['water_heater_names']}")
    print(f"HVAC heating schedules: {inspection['hvac_heating_schedules'][:3]}")
    print(f"HVAC cooling schedules: {inspection['hvac_cooling_schedules'][:3]}")
    print(f"DHW setpoint schedules: {inspection['dhw_setpoint_schedules'][:3]}")

    monitor_path = Path(args.monitor)
    monitor_df = pd.read_csv(monitor_path)

    ts = prepare_hvac_dhw_timeseries(monitor_df)
    comfort = ComfortConfig(
        hvac_occ_band=(args.occ_min, args.occ_max),
        dhw_band=(args.dhw_min, args.dhw_max),
    )
    envelope = compute_comfort_priority_flex_envelope(ts, comfort=comfort)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    envelope.to_csv(out_csv, index=False)

    out_gams = Path(args.out_gams)
    out_gams.parent.mkdir(parents=True, exist_ok=True)
    envelope_to_gams_inc(envelope, out_gams)

    print(f"Saved envelope CSV: {out_csv}")
    print(f"Saved GAMS include: {out_gams}")


if __name__ == "__main__":
    main()
