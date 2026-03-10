# HVAC + DHW CLUE adaptation notes

## 1) Building inspection result needed for control adaptation

Use `inspect_building_hvac_dhw(...)` to extract:

- HVAC heating/cooling setpoint schedules (`ThermostatSetpoint:DualSetpoint`),
- DHW setpoint schedules (`WaterHeater:Mixed`),
- Water heater object names.

Supported model formats:
- `.idf` (parsed directly),
- `.epJSON`.

These are then mapped into Sinergym `action_definition` so CLUE can control
both systems:

- HVAC heating setpoint action,
- HVAC cooling setpoint action,
- DHW setpoint action.

## 2) Comfort-first flexibility envelope definition

For each timestep `t`, compute:

- `P_BASE(t)` = baseline HVAC + DHW power,
- `P_MIN(t)` = lowest feasible power without violating comfort,
- `P_MAX(t)` = highest feasible power (pre-heating / pre-cooling / DHW charging) without violating comfort.

The generated output has:

- `p_base_kw`, `p_min_kw`, `p_max_kw`,
- `flex_down_kw`, `flex_up_kw`,
- HVAC and DHW comfort violation indicators.

## 3) GAMS handoff

`envelope_to_gams_inc(...)` exports:

- set `t`,
- parameters `P_BASE(t)`, `P_MIN(t)`, `P_MAX(t)`.

This can be included directly in a GAMS model for dispatch or bidding
constraints:

`P_MIN(t) =L= P_DECISION(t) =L= P_MAX(t)`.

## 4) Recommended advanced algorithms for this application

For better performance than plain random-shooting CLUE:

1. **Chance-constrained MPC + GP dynamics**
   - Deterministic optimization with uncertainty margins.
   - Directly enforces comfort probability constraints.
2. **Distributionally Robust MPC (DR-MPC)**
   - Protects against forecast/model distribution shift.
   - Good for occupancy and DHW demand uncertainty.
3. **Safe RL with Control Barrier Functions (CBF-RL)**
   - Learns aggressively while safety filter enforces comfort.
4. **Constrained SAC/PPO with Lagrangian dual updates**
   - Scales better for high-dimensional multi-zone control.
5. **Hierarchical control (upper envelope optimizer + lower tracking MPC)**
   - Upper layer negotiates flexibility envelope;
   - Lower layer tracks setpoints while respecting comfort.
