# Attack Strategy Benchmark (Optional Comparison Study)

**NOTE:** This study is OPTIONAL. The canonical project remains frozen on the `main` branch. This repository extension explores comparative attack efficiencies.

## Executed Attacks
- Random
- MCMC
- Evolutionary
- Attribution-Guided (High, Low, Random)

## Blocked / Optional Attacks
- Generative (IMPLEMENTED_NOT_EXECUTED - compute constraint)
- RL (BLOCKED_COMPUTE)
- LLM-guided (BLOCKED_EXTERNAL_CREDENTIALS)

## Reproducing
Install `torch`, `pandas`, `numpy`, `matplotlib`, `scipy`.
Run `python scripts/run_benchmark.py`

Results are stored in `results/`.
