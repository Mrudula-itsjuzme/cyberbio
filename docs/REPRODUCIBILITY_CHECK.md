# FINAL REPRODUCIBILITY AUDIT

## System Environment
- Environment correctly initialized using `source .venv/bin/activate`.
- All Python dependencies explicitly declared in project setup.

## Script Audits
1. **Tests:** Full suite passed via `PYTHONPATH=. pytest tests` (429 passed, 1 skipped).
2. **Smoke Test:** Executed successfully via `PYTHONPATH=. python scripts/smoke_test_attack.py`, producing valid prediction drifts in eV.
3. **Multi-seed Validation:** Executed successfully via `PYTHONPATH=. python scripts/run_multiseed.py`. Extracted 3-seed multi-run metrics, saved natively to `results/final/` as `.csv` matrices.
4. **Master Script:** `scripts/run_final_experiments.py` successfully orchestrates the entire reproducible loop via `--smoke`, `--multiseed`, and `--full` flags.

All documented commands run without manual intervention.

---

