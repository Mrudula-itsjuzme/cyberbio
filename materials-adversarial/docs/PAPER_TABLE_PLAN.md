# Paper Table Plan

## Table 1: Dataset and Split Summary
*   **Columns:** Total Samples, Train Split, Validation Split, Exposed Test Split, Target Property Unit, Target Scaling.
*   **Content:** 4,209 samples, exact seed (if applicable), eV unit, and a footnote stating the test set is "exposed."

## Table 2: Model Architectures and Parameter Counts
*   **Columns:** Model Name, Architecture Type, Parameter Count, Training Goal.
*   **Content:** Ordinary Transformer (~12M), Architecture Control (~12M), Mixed-robust Transformer (~12M), Augmented Transformer (~12M), GraphMPNN (~5M).

## Table 3: Canonical Performance/Robustness Metrics
*   **Columns:** Model, Clean Validation MAE (eV), RMSE, R², Equivalent-SMILES Drift (eV).
*   **Content:** Exact metrics pulled directly from `docs/FINAL_RESULTS.md`.

## Table 4: Attack Definitions
*   **Columns:** Attack Class, Operation Type, Representation Changed?, Chemistry Changed?, Original Label Valid?.
*   **Content:** Equivalent SMILES (No, No, Yes), Substitution (Yes, Yes, No), Deletion (Yes, Yes, No).

## Table 5: Adaptive Search Results
*   **Columns:** Search Strategy, Edit Budget, Query Budget, Mean Max Drift (eV), Absolute Repaired Max (eV).
*   **Content:** Q10/Q20/Q50 results for Random/Greedy/MCMC under <=3 edits, highlighting the 3.19 eV repaired maximum.

## Table 6: Claims Ledger
*   **Columns:** Claim, Supported By, Status (Verified / Rejected / Blocked).
*   **Content:** Direct mapping to `docs/FINAL_PROJECT_STATUS_TABLE.md`, specifically showing the rejection of two-branch semantics and the blocked status of oracle validation.

## Table 7: Limitations and Unresolved Physical Validation
*   **Columns:** Limitation, Implication, Required Future Action.
*   **Content:** Exposed test set, unknown original DFT geometry, unexecuted HPC surrogate calibration, black-box search limitations.
