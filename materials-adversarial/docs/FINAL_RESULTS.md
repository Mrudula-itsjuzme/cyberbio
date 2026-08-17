# Final Results Table

The following table presents the authoritative results of the Phase 2 comparative evaluation exactly as extracted from the raw metric files.

| Evaluation Metric | Baseline Model | Defended Model | Interpretation |
| :--- | :--- | :--- | :--- |
| **Clean MAE** | 0.4619 eV | 0.4601 eV | The defense preserved and marginally improved clean accuracy on the sealed test set. |
| **Substitution Success** | 20.13% | 7.53% | Highly successful local defense against seen attack types. |
| **Rearrangement Success** | 9.29% | 2.83% | Highly successful local defense against seen attack types. |
| **Insertion Success** | 32.61% | 32.46% | Complete failure to generalize defense to unseen structural alterations. |
| **Deletion Success** | 32.13% | 30.04% | Complete failure to generalize defense to unseen structural alterations. |
| **Mean Adv. Drift** | 0.4655 eV | 0.4290 eV | Overall adversarial sensitivity decreased slightly across the aggregate valid test set. |
| **Max Adv. Drift** | 5.5894 eV | 5.5167 eV | Extreme vulnerability outliers remain present regardless of defense. |

*Note: All values were evaluated over the identical sealed 10% Test Split using symmetrical adversarial candidate strings generated with `attack-seed = 42`.*
