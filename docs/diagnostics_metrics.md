# Diagnostics & Metrics

The system tracks a variety of metrics to ensure simulation health and provide "proxy loss" signals for future surrogate training.

## Simulation Metrics (`sim/`)
- `sim/temperature_min`: Minimum temperature in the domain.
- `sim/temperature_max`: Maximum temperature (peak).
- `sim/temperature_mean`: Mean temperature.
- `sim/temperature_std`: Standard deviation of temperature.

## Stability Metrics (`stability/`)
Used to detect numerical explosions or invalid states.
- `stability/nan_count`: Number of NaN values (should be 0).
- `stability/inf_count`: Number of Inf values (should be 0).
- `stability/max_abs_dT`: Maximum absolute change in temperature in one step.
- `stability/update_ratio`: Ratio of `max_abs_dT` to `max_abs_T`. High values indicate instability.

## Energy & Proxy Losses (`energy/`, `solver/`)
These metrics serve as physical consistency checks and proxy training objectives.
- `energy/power_in_W`: Instantaneous power input from laser.
- `energy/energy_in_J`: Cumulative energy input.
- `energy/temperature_l1`: L1 norm of temperature field.
- `energy/temperature_l2`: L2 norm of temperature field.
- `solver/residual_proxy`: Relative change magnitude `||T_new - T_old|| / ||T_old||`.

## Performance (`perf/`)
- `perf/step_time_ms`: Wall time per simulation step.
- `perf/steps_per_sec`: Throughput.
- `perf/gpu_mem_alloc_mb`: GPU memory allocated.
