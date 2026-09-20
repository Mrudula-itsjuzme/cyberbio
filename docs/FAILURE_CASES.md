# FAILURE CASE ANALYSIS

## Strongest Successful Adversarial Example (Baseline Model)
- **Sample ID:** `poly_4102`
- **Original Sequence:** `[*]C[*]`
- **Adversarial Sequence:** `*C*`
- **Validity Status:** `valid`
- **True Target:** 6.8063 eV
- **Original Prediction:** 7.1652 eV
- **Adversarial Prediction:** 2.4368 eV
- **Absolute Drift:** 4.7284 eV
- **Attack Family:** Randomization
- **Source File:** `materials-adversarial/results/phase2_paired_benchmark/clean_attack_records.jsonl`

## Weakest Adversarial Example / Failure to Attack
- **Sample ID:** `poly_3709`
- **Original Sequence:** `[*]CCCCCCOC(=O)NCCCCCCCCCCNC(=O)O[*]`
- **Adversarial Sequence:** `[*]CCCCCCOC(=O)NCCCCCCCCCNNC(=O)O[*]`
- **Validity Status:** `valid`
- **True Target:** 6.2268 eV
- **Original Prediction:** 5.5640 eV
- **Adversarial Prediction:** 5.5640 eV
- **Absolute Drift:** 0.0000 eV
- **Attack Family:** Substitution
- **Source File:** `materials-adversarial/results/phase2_paired_benchmark/clean_attack_records.jsonl`
