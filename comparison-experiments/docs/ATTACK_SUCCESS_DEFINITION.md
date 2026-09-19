# ATTACK SUCCESS DEFINITION

The repository contains multiple configured definitions for adversarial success across different scripts and phases. The underlying requirement is typically a combination of positive `prediction_drift` and constraint satisfaction, but the explicit threshold ($\tau$) varies.

## Provenance of Definitions

1. **Plot Budget Sweep (`comparison-experiments/scripts/plot_budget_sweep.py:18-19`)**
   - Config Key/Logic: `is_success`
   - Value: `prediction_drift > 0.05` AND `constraint_pass == True`
   - Units: Scaled target property units.

2. **Metrics Statistics (`comparison-experiments/src/metrics/statistics.py:16-17`)**
   - Config Key/Logic: `successes`
   - Value: `prediction_drift > 0.1` AND `constraint_pass == True`
   - Experiment(s): Baseline classical metric aggregations.

3. **Phase 1 Utilities (`Project/phase1/attack_engine.py:678`)**
   - Config Key/Logic: `drift_threshold` parameter in `summarise_attack_results`
   - Value: Defaults to `0.5`, overridden to `0.1`.
   
4. **Phase 1 Output (`Project/outputs/phase1_results.json:17`)**
   - Config Key/Logic: `"drift_threshold"`
   - Value: `0.3`

## Conclusion
For classical attack comparison aggregation, we will explicitly report success rates at multiple configured thresholds ($\tau = 0.05$ and $\tau = 0.1$) accompanied by `constraint_pass == True` rather than relying on a single hardcoded threshold or assuming any `drift > 0` is successful.
