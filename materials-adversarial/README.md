# CyberBio: Materials Adversarial Robustness

## Research Objective
This project evaluates the adversarial robustness of Transformer-based sequence models used for materials property prediction (specifically polymer bandgap, $E_g$, in eV) against representation-preserving and chemistry-changing perturbations.

## Core Setup
- **Task:** Bandgap regression in eV.
- **Model:** Two-Branch Transformer Regressor.
- **Attack Classes:**
  1. *RandomSMILES*: Generates alternative valid PSMILES representations (preserves exact chemistry).
  2. *MCMC*: Generates localized token edits restricted by Tanimoto similarity and heuristic checks (chemically changing).
- **Defense Classes:**
  1. *RandSmilesAug*: Data augmentation using randomized representations.
  2. *MCMCDefended*: Adversarial consistency training.
  3. *CombinedDefense*: Full hybrid defense.

## Verified Key Results
- TransPolymer is brittle to both alternate sequence representations and localized topological edits.
- Data augmentation reduces representation drift and acts as an out-of-distribution regularizer (improving clean Scaffold Split RMSE).
- Defenses are orthogonal; the CombinedDefense performs best across all evaluated metrics.
- Multi-seed validation (N=3) confirms that the vulnerability and defense improvements are highly stable across initializations.

## Reproduction
To replicate the final verified results, including multi-seed validation and analysis matrix generation:
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_final_experiments.py --full
```
*Note: Run `pytest tests` to verify the full RDKit and tokenization framework.*

## Artifacts and Documentation
- **Final Results / Tables:** `results/final/`
- **Full Research Narrative:** `docs/FINAL_REPORT.md`
- **Viva / Study Materials:** `docs/VIVA_STUDY_GUIDE.md`, `docs/PROJECT_CHEATSHEET.md`

## Limitations
The primary limitation of this study is the lack of a Density Functional Theory (DFT) physical oracle. The MCMC perturbations successfully probe the model's topological sensitivity under chemically constrained edits, but we do not claim that the generated variants perfectly preserve the true physical bandgap label.
