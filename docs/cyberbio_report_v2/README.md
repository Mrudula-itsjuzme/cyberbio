# CyberBio Report V2

Audited documentation for the CyberBio project, reconstructed against the repository state and extended with the September 21, 2026 scientific-integrity and representation-robustness review.

## Reading order

0. [Complete Master Report](CyberBio_V2_Master_Report.md)
1. [Project Overview V2](CyberBio_Project_Overview_V2.md)
2. [Part I — Foundations, Transformer, and GraphMPNN](CyberBio_Technical_Report_Part1_V2_Foundations_Transformer_GraphMPNN.md)
3. [Part II — Attacks, Search, Defenses, and Oracle](CyberBio_Technical_Report_Part2_V2_Attacks_Search_Defenses_Oracle.md)
4. [Part III — Experiments, Audits, and Evidence](CyberBio_Technical_Report_Part3_V2_Experiments_Audits_Evidence.md)
5. [September 21 Representation-Robustness Review](../../materials-adversarial/docs/REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md)

## Evidence policy

The report distinguishes:

- **measured result**;
- **interpretation**;
- **hypothesis**;
- canonical evidence;
- historical evidence;
- rejected evidence;
- blocked claims;
- developmental-unverified work.

The September 21 extension adds the validation-only multi-view negative result, corrected chemistry-changing label semantics, fail-closed graph parsing, deterministic split behavior, molecule-level bootstrap analysis, disagreement/error diagnostics, and the current 448-passed test state.

In particular, do not convert:

- validation findings into sealed-test claims;
- prediction drift into physical error;
- disagreement correlations into causal or calibrated-uncertainty claims;
- one-checkpoint architecture contrasts into multi-seed generalizations.

## GitHub-native source of truth

These Markdown files are the editable GitHub-native source. Generated DOCX/PDF editions should be regenerated after documentation changes rather than treated as independent scientific sources of truth.
