# Reproducibility Guide & Verification Audit

## 1. Reproducibility Guarantee

This project enforces strict reproducibility across all training, attack generation, evaluation, and plotting steps. All experimental numbers reported in the thesis report and paper manuscript are generated programmatically via automated scripts.

---

## 2. Environment Setup & Dependency Hashes

### System Environment
- **Python**: Version 3.12.3.
- **PyTorch**: Version 2.13.0+cpu.
- **RDKit**: Version 2026.03.5.
- **Virtual Environment**: `.venv` located in workspace root.

### Environment Setup Commands
```bash
# Navigate to project directory
cd materials-adversarial

# Activate virtual environment
source .venv/bin/activate

# Install requirements (if resetting environment)
pip install -r requirements.txt
```

---

## 3. Step-by-Step Execution Protocol

### Step 1: Run Full Unit Test Suite (430 Tests)
```bash
PYTHONPATH=.:src .venv/bin/pytest tests
```
*Expected Output*: `429 passed, 1 skipped, 0 failures`.

### Step 2: Execute Multi-Seed Benchmark Suite (5 Seeds)
```bash
PYTHONPATH=.:src .venv/bin/python scripts/run_comprehensive_benchmark_suite.py
```
*Outputs*: Generates `results/comprehensive_benchmark_summary.json` containing 5-seed statistics ($42, 123, 2026, 777, 999$).

### Step 3: Generate Publication-Quality Figures
```bash
PYTHONPATH=.:src .venv/bin/python scripts/generate_publication_plots.py
```
*Outputs*: Saves PNG plots to `outputs/`:
- `outputs/baseline_vs_defended_multiseed.png`
- `outputs/mcmc_steps_drift_curve.png`
- `outputs/ablation_study_chart.png`

### Step 4: Execute External Reproducibility Audit
```bash
PYTHONPATH=.:src .venv/bin/python scripts/reproducibility_audit.py
```
*Expected Output*: Logs `Reproducibility Audit Complete: System Verified`.

---

## 4. Frozen Artifact Manifest

| Artifact File | Path | Description |
| :--- | :--- | :--- |
| **Benchmark Summary JSON** | `results/comprehensive_benchmark_summary.json` | Master multi-seed metrics artifact |
| **Figure 1 PNG** | `outputs/baseline_vs_defended_multiseed.png` | Multi-seed robustness bar chart |
| **Figure 2 PNG** | `outputs/mcmc_steps_drift_curve.png` | Attack budget sensitivity plot |
| **Figure 3 PNG** | `outputs/ablation_study_chart.png` | Lambda & Tanimoto ablation chart |
| **Release Manifest** | `RELEASE_MANIFEST.json` | Checkpoint and scaler SHA-256 hashes |
