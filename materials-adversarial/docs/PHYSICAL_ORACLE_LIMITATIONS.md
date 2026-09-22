# Physical-oracle limitations and blockers

## Resolved provisioning blockers

1. QE 7.6 is installed locally with executable hashes and build provenance.
2. SSSP 1.3.0 PBE Precision is verified for all nine required elements.
3. Initial cutoffs are derived from present-element SSSP maxima and enforced fail-closed.
4. The backend now emits complete isolated-cell inputs and parses finite molecular orbital gaps.
5. Signed model predictions are recovered separately from oracle results.

## Remaining hard blockers to scientific interpretation

1. Cutoff, vacuum, SCF-threshold, repeat-length, and relaxation convergence are unexecuted. The frozen 12 angstrom minimum-vacuum plan is outside the local host memory envelope even for a smaller 22-atom C/H structure.
2. No local batch scheduler or known remote allocation is available from this checkout.
3. The historical polyVERSE electronic-structure protocol is not documented locally.
4. PSMILES does not uniquely determine periodic translation, torsion, packing, or cell.
5. The train/validation calibration and all paired oracle calculations remain unexecuted.

## Scope limitations

- The proposed oracle is a finite, hydrogen-capped oligomer surrogate. It is not a periodic bulk polymer calculation.
- PBE Kohn-Sham gaps are protocol-dependent electronic quantities and should not be described as experimental optical gaps.
- The calibration set is small and deliberately excludes the sealed test split.
- Force-field conformer generation is a starting geometry heuristic; conformational uncertainty is not yet quantified.
- Six of 36 requested calibration structures failed construction, including longer oligomers for three source molecules. Missingness must remain visible.
- Four retained attack pairs passed conservative graph/charge plausibility checks, but no DFT structure has been generated and chemical synthesis plausibility has not been established.
- Absolute target agreement and paired-change reliability are separate questions. Good rank correlation alone cannot validate `delta_T`.

## Evidence labels

- **MEASURED** means directly observed in this checkout or generated artifact.
- **INTERPRETATION** means a conclusion drawn from those observations.
- **HYPOTHESIS** means a proposition awaiting the frozen experiment.
- **BLOCKED** means the needed evidence does not exist or cannot be generated in the present environment.

No blocked quantity should be represented as zero, negative evidence, or a completed experiment.
