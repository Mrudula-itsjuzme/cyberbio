# Paper Limitations

* **Exposed Test Set**: The test set is not pristine, as it was repeatedly queried during iterative metric evaluations and framework debugging.
* **Incomplete Dataset Provenance**: The source dataset's exact DFT physics protocol is incomplete, and the original 3D periodic geometries are unavailable, forcing reliance on 2D topological mapping.
* **Lack of Ground-Truth Chemistry Edits**: Real physical bandgaps (the "truth") for the computationally generated chemistry edits are unavailable, leaving the physical validity of the adversarial perturbations strictly unverified.
* **Unexecuted Surrogate Calibration**: The transparent 9-job surrogate QC calibration pilot has been strictly prepared but remains unexecuted due to a lack of local HPC/SLURM access.
* **Partial Chemical Edit Space**: The bounded mutation operators (substitutions and deletions) cover only a subset of the physically valid chemical edit space.
* **Missing Insertion Experiments**: While equivalent representations were tested, there is no canonical experiment evaluating strictly constrained structural insertion.
* **Heuristic Search Scope**: The adversarial search is black-box, greedy, and discrete; it is not mathematically exhaustive.
* **Task Specificity**: The GraphMPNN evaluation is dataset- and task-specific (polyVERSE bandgap regression). We make no claim of universal chemistry robustness for this architecture.
