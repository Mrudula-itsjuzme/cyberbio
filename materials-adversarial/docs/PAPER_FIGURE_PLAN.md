# Paper Figure Plan

## Figure 1: Full Framework Architecture
*   **Purpose:** Illustrate the generic, domain-agnostic architecture separating RepresentationAdapter, AttackOperator, ConstraintSet, ValidityChecker, and the Predictor vs. Oracle evaluation.
*   **Data Source:** Conceptual diagram.
*   **Plot Type:** Flowchart / Diagram.
*   **Source:** `docs/GENERAL_ADVERSARIAL_FRAMEWORK.md` (to be rendered graphically).

## Figure 2: Transformer vs. GraphMPNN Representation Sensitivity
*   **Purpose:** Visually demonstrate the 0.614 eV representation drift of the sequence model compared to the ~0 eV drift of the GraphMPNN on identical molecules.
*   **Data Source:** `docs/FINAL_RESULTS.md` / `results/representation_attribution_sequential_sample`.
*   **Plot Type:** Bar chart or Box plot comparing representation prediction variance per molecule.
*   **Source:** Generated via a matplotlib script querying the canonical metric logs.

## Figure 3: Canonical Model Comparison
*   **Purpose:** Compare the clean validation MAE and overall robustness tradeoffs between the 5 architecture variants.
*   **Data Source:** `docs/FINAL_RESULTS.md`.
*   **Plot Type:** Scatter plot (Clean MAE vs. Equivalent-SMILES drift) with marker size indicating parameter count.
*   **Source:** `docs/FINAL_RESULTS.md`.

## Figure 4: Substitution/Deletion Stress
*   **Purpose:** Show the distribution of prediction drifts when random valid substitutions and deletions are applied to the GraphMPNN.
*   **Data Source:** `docs/FINAL_RESULTS.md` / `results/phase4_controlled_robustness`.
*   **Plot Type:** Histogram or Violin plot of prediction drift (eV) separated by operator type.
*   **Source:** Log files from single-edit stress tests.

## Figure 5: Adaptive Search Drift vs. Query Budget
*   **Purpose:** Demonstrate how increasing the heuristic search budget (Q10, Q20, Q50) increases the discovered maximum prediction drift within the strict <=3 edit bound.
*   **Data Source:** `results/phase7_adaptive_search` / `results/phase8_multi_edit_search`.
*   **Plot Type:** Line plot (mean max drift across test set) with error bands over query budgets for Random vs. Greedy vs. MCMC.
*   **Source:** Outputs from adaptive search scripts.

## Figure 6: Cross-Model Transfer
*   **Purpose:** Show that adversarial candidates discovered on the Transformer frequently induce significant, aligned drift on the GraphMPNN (and vice versa).
*   **Data Source:** Transferability experiments (Phase 5).
*   **Plot Type:** Correlation scatter plot of `Delta_G_ModelA` vs `Delta_G_ModelB` for shared adversarial candidates.
*   **Source:** `results/phase5_deletion_transfer` logs.

## Figure 7: Attack→Oracle Conceptual Loop
*   **Purpose:** Highlight the gap between ML prediction drift and physical truth, mapping out the `CALIBRATABLE_SURROGATE` protocol up to the HPC boundary.
*   **Data Source:** Conceptual.
*   **Plot Type:** Diagram outlining surrogate geometry generation (ETKDG) passing to Quantum ESPRESSO.
*   **Source:** Narrative from `docs/PROJECT_STORY.md`.

## Figure 8: Project Evolution / Decision Map
*   **Purpose:** Summarize the negative results, rejected hypotheses (two-branch), and bug fixes (inflation bug) that drove the project's scientific evolution.
*   **Data Source:** `docs/PROJECT_STORY.md` / `docs/FINAL_PROJECT_STATUS_TABLE.md`.
*   **Plot Type:** Directed acyclic graph or timeline flowchart.
*   **Source:** High-level project audit logs.
