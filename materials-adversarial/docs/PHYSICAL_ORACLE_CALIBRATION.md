# Physical-oracle calibration plan

Status: **PREPARED; NO DFT OBSERVATIONS**

## Frozen set

- **MEASURED:** 12 neutral polymers were selected deterministically from train/validation only: four each from low, middle, and high dataset-gap tertiles.
- **MEASURED:** selection combines target tertiles with fingerprint farthest-point sampling. Eleven records are from train and one from validation; zero are from the sealed test split.
- **MEASURED:** repeat counts 2, 3, and 4 were attempted. Thirty SDF structures succeeded and six failed. Failures remain explicit in `structure_failures.csv`.
- **INTERPRETATION:** this small set probes chemical and target diversity but is not a statistically complete validation population.

The exact membership, original/canonical PSMILES, split, target, charge, elements, and selection basis are frozen in `results/physical_oracle_phase/20260921/calibration_manifest.csv`.

## Convergence before calibration

One successfully constructed repeat-count-3 structure from each target stratum is used for the convergence matrix. For each, test:

- wavefunction cutoff multiplier: 1.00, 1.25, 1.50;
- vacuum: 12, 15, 18, 22 angstrom;
- SCF threshold: 1e-6, 1e-8, 1e-10 Ry.

The plan contains 30 blocked calculations. Each parameter sweep changes one factor from the ultimately declared baseline. Full curves, failures, wall time, core-hours, and storage must be retained.

## Calibration analysis

For each successful repeat count, report the raw dataset target and oracle gap without re-fitting either quantity. Report:

- coverage and failure rate;
- MAE, RMSE, median absolute error, and signed bias;
- Pearson and Spearman association;
- least-squares slope and intercept, clearly labelled descriptive;
- residuals versus target stratum, elements, molecular size, and oligomer length;
- rank stability across repeat counts;
- paired `n=2 -> 3` and `n=3 -> 4` changes where both structures succeed.

No arbitrary affine correction or threshold will be chosen after seeing the results. If a mapping is later fitted, it must be selected on a further train-only partition and assessed on the frozen validation records.

## Predeclared interpretation rubric

- **Unsuitable:** widespread execution/structure failures, unstable oligomer-length ordering, or no useful signed/rank relationship.
- **Exploratory only:** qualitative rank information exists, but absolute or differential errors remain too unstable for attack adjudication.
- **Usable for differential checks:** signed paired changes are stable enough to test direction and discrepancy, even if absolute targets are biased.
- **Strong calibration:** both absolute and differential behavior are stable across the frozen checks. This still does not establish identity with the historical polyVERSE protocol.

These are interpretation categories, not numerical acceptance thresholds. The observed metrics and uncertainty must drive the classification transparently.

## Present result

- **MEASURED:** structure preparation completed as described above.
- **BLOCKED:** convergence metrics, calibration metrics, rank correlation, and oligomer-length behavior are unavailable because the frozen 12 angstrom convergence floor exceeds the local memory envelope. QE 7.6 and SSSP 1.3.0 PBE Precision are provisioned.
- **BLOCKED:** the calibration gate has not passed; paired attack evaluation is prohibited.
