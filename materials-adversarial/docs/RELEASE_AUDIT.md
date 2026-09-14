# Attack-Family Status Audit

The following table clarifies the evaluation status of various attack families mentioned throughout the project.

| Attack Family | Status | Description |
| :--- | :--- | :--- |
| Equivalent-SMILES (Representation) | CANONICALLY_EVALUATED | Evaluated on Transformer (0.614 eV drift) and GraphMPNN (0.0 eV drift). |
| Single-edit Substitution | CANONICALLY_EVALUATED | Used for fixed-bank stress testing. |
| Single-edit Deletion | CANONICALLY_EVALUATED | Used for deletion transfer stress testing. |
| Single-edit Insertion | NOT_TESTED | Mentioned in README as possible chemistry-changing edit, but not canonically evaluated in fixed banks or search. |
| Bounded $\le 3$ Multi-Substitution Adaptive Search | CANONICALLY_EVALUATED | Phase 12B established the ~3.19 eV drift bound under this setting. |
| Unbounded Adaptive Search | NON-CANONICAL | Phase 12 originally inflated edit budgets causing 5.961 eV drift due to fragmentation. |
