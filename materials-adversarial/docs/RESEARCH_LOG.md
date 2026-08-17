# Research Log: Materials Adversarial Learning

This log documents every major decision made during the construction of the unified adversarial framework for robust materials sequence modeling.

## 1. Dataset Selection
- **WHAT we did**: Selected the polyVERSE Bandgap dataset (~4,200 records) over OpenPoly or PI1M for supervised training.
- **WHY we did it**: PI1M lacks experimental property labels. OpenPoly possesses too few records per property (e.g., <400 valid for Tg), leading to extreme shortcut learning and overfitting. polyVERSE provides thousands of clean, syntactically homogenous (PSMILES) experimental/DFT measurements.
- **HOW it was implemented**: Fetched the `bandgap_chain.csv` file, updated `configs/dataset.yaml`, and mapped it into the internal ETL schema (`original_representation` and `property_value`).
- **WHAT assumption it relies on**: Assumes the representation (PSMILES) inherently captures the microstructural properties responsible for macroscopic bandgap without needing 3D geometry.
- **WHAT could go wrong**: The model might memorize trivial length-based shortcuts rather than learning chemical semantics.
- **HOW we measured whether it worked**: We evaluated the trained Transformer baseline on a strictly held-out test set, achieving an MAE of 0.4619 eV and R² of 0.8019.

## 2. RDKit Validity as Plausibility
- **WHAT we did**: Enforced strict RDKit validation (`Chem.MolFromSmiles(seq) is not None`) before counting any generated adversarial sequence as a successful attack.
- **WHY we did it**: Machine learning models often break when fed out-of-distribution junk strings. If an attack generates chemically impossible valencies (e.g., Carbon with 5 bonds), the model's prediction drift represents a failure to handle corrupted inputs, not an adversarial vulnerability in material space.
- **HOW it was implemented**: In `run_attacks.py`, every candidate variant is parsed by `materials_adv.utils.chemistry.is_valid_smiles`. Invalid candidates are recorded but disqualified from success calculations.
- **WHAT assumption it relies on**: Assumes that RDKit's SMILES parser accurately distinguishes between physically realizable polymer repeat units and physically impossible ones.
- **WHAT could go wrong**: RDKit might reject highly exotic but real polymers, or accept unstable intermediates that cannot exist as bulk materials.
- **HOW we measured whether it worked**: Tracked the valid vs. invalid rejection rate (30.6% rejected overall, ~50% rejected for Insertion/Deletion).

## 3. Generic Property Architecture
- **WHAT we did**: Decoupled the Transformer regression objective from "Tg" to a generic generic "Target Property" dynamically loaded at runtime.
- **WHY we did it**: To prove the adversarial vulnerabilities are a flaw in representation-learning paradigms, not an artifact of Tg-specific physics.
- **HOW it was implemented**: Rewrote `records.py` schema, `dataset.yaml`, and `train.py` to map properties natively via `property_value` and scale them based on standard score scalers.
- **WHAT assumption it relies on**: The base Transformer architecture is expressive enough to model various properties (thermal, electronic, etc.) without domain-specific inductive biases.
- **WHAT could go wrong**: Fixed hyperparameters might not translate well across domains with vastly different scales (e.g., K vs eV).
- **HOW we measured whether it worked**: Successfully achieved high predictive accuracy (R² > 0.80) on Bandgap without modifying the core model architecture.

## 4. Adversarial Threat Model Definition
- **WHAT we did**: Defined attack success strictly as `absolute_prediction_drift > Test_MAE (0.4619 eV)` applied exclusively to valid/plausible sequences.
- **WHY we did it**: To establish a rigorous lower bound. An adversarial perturbation must trick the model *worse* than the model's average expected error on clean data.
- **HOW it was implemented**: Calculated drift dynamically in the attack generator, storing it in `phase1e_results.jsonl`.
- **WHAT assumption it relies on**: Assumes the test MAE is a representative baseline of standard model uncertainty.
- **WHAT could go wrong**: If the model is exceptionally poor, the MAE bound is too loose.
- **HOW we measured whether it worked**: Verified that nearly 25% of valid generated attacks successfully breached this bound, confirming systemic vulnerability.

## 5. Phase 2 Augmentation Policy (Label Preservation)
- **WHAT we did**: Selected only Substitution and Rearrangement attacks (with budget=1) to construct the adversarial training set, assigning them their original experimental property label.
- **WHY we did it**: Insertions and Deletions can fundamentally alter stoichiometry and bandgap (e.g., adding a Nitro group). We cannot safely assume large structural edits preserve the ground truth.
- **HOW it was implemented**: Filtered the 79,825 training attacks via `is_label_preserving()` logic in `augment_dataset.py`, adding exactly 11,300 eligible records to the train pool.
- **WHAT assumption it relies on**: Assumes a single isosteric substitution or sequence reordering on a polymer repeat unit does not shift the true physical Bandgap beyond standard experimental noise thresholds.
- **WHAT could go wrong**: We might accidentally teach the model to predict the *wrong* bandgap for a sequence that actually *did* undergo a physical phase/electronic shift due to that single substitution.
- **HOW we measured whether it worked**: Comparative Phase 3 Evaluation (Baseline vs Defended on holdout adversarial tests).
