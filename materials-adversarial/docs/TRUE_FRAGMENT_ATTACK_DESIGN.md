# True Fragment / Subgraph Replacement Attack — DESIGN ONLY

**Status: DESIGN ONLY. NOT IMPLEMENTED. NO RESULTS EXIST.**

This note exists because the forensic audit found that the operator previously named
`FunctionalGroupReplacementAttack` does not manipulate functional groups at all
(`docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md` section 10): it substitutes one aliphatic carbon
at a time with `O`, `N`, `F`, and — rarely — `C#N` / `C(F)(F)F`. It has been renamed
`AliphaticCarbonSubstitutionAttack`. Until a real fragment operator exists, no
"functional-group replacement" claim may appear anywhere in the paper.

---

## 1. What "fragment replacement" has to mean

A genuine fragment-replacement operator deletes a **connected subgraph** from the polymer
and grafts a **different connected subgraph** in its place, preserving the substitution
count at the cut. The unit of an application is the swapped fragment, not an atom.

The scientific motivation is different from atom substitution. Atom substitution perturbs
local electronic structure; fragment replacement changes which functional chemistry is
present while keeping the skeleton, which is the closest SMILES-space analogue of a real
synthetic modification.

## 2. Fragment library

A fragment is a SMILES with exactly two labelled attachment points, because both cut
bonds must be re-made:

```
[1*]-<fragment core>-[2*]
```

Proposed families (curated, small, chemically ordinary — no exotic groups at first):

| family | example fragment (`[1*]…[2*]`) | note |
|---|---|---|
| ester ↔ amide ↔ thioester | `[1*]C(=O)O[2*]`, `[1*]C(=O)N[2*]`, `[1*]C(=O)S[2*]` | isosteric, valence-safe |
| ether ↔ thioether ↔ amine | `[1*]O[2*]`, `[1*]S[2*]`, `[1*]N[2*]` | very common in this dataset |
| carbonyl → thiocarbonyl | `[1*]C(=S)[2*]` | single-atom valence change |
| sulfone / sulfoxide | `[1*]S(=O)(=O)[2*]`, `[1*]S(=O)[2*]` | matches existing scaffolds |
| alkane linker length | `[1*]CC[2*]`, `[1*]CCC[2*]`, `[1*]CCCC[2*]` | length change, no heteroatom |
| aromatic linker | `[1*]c1ccccc1[2*]` | ring introduction |
| fluorinated block | `[1*]C(F)(F)[2*]`, `[1*]C(F)(F)F` (mono-attached) | mono-attached form needs its own rule, see §4 |
| nitrile-bearing | `[1*]C(C#N)[2*]` | adds a pendant group, not a cut |

A mono-attached fragment (one attachment point) grows the graph; a di-attached fragment
keeps the backbone intact. Both are legitimate but they are **different edit types** and
must be reported separately.

## 3. Attachment rules

1. **Source cut selection.** Choose two atoms `(a, b)` in the source that are both
   heavy, both have ≥ 1 heavy neighbour after the cut, and — the important constraint —
   are connected by a path that does **not** contain a polymer attachment point (`[*]`)
   or a ring bond. Cutting a ring bond changes the ring count; that is a different,
   larger edit and is excluded in v1.
2. **Wildcard protection.** `[*]` atoms and the two bonds adjacent to each are never cut,
   never replaced, and their count must be preserved. This is enforced, not assumed: the
   candidate must satisfy `require_attachment_count == count([*] in source)`.
3. **Scaffold interaction.** `ScaffoldPreservingAttack` may wrap this operator unchanged
   (`ScaffoldPolicy` checks subgraph retention of the source Murcko scaffold), but the
   natural combination is a *scaffold-altering* fragment swap, which the scaffold wrapper
   will reject. Decide explicitly which question is being asked before composing them.
4. **No-op rejection.** If the grafted fragment is identical to the removed one, the
   candidate is rejected and must not consume a query.

## 4. Bond mapping and valence

* Cut bonds `(a–x)` and `(b–y)` are broken; the two new bonds are `(x–f1)` and `(y–f2)`,
  where `f1`, `f2` are the fragment's attachment atoms.
* Bond order at the graft is taken from the fragment's attachment bonds (single by
  default). Restoring the original bond orders is only correct for isosteric swaps and
  must not be assumed in general.
* **Valence check is mandatory after every graft** (`Chem.SanitizeMol`); a rejected
  valence must drop the candidate, never be repaired silently.
* Formal charge, aromaticity and implicit-hydrogen counts must be compared before/after
  and reported (`domain/chemistry/graph_metrics.py` already computes element, atom, bond
  and component deltas canonically).
* **Connectivity**: `require_single_component=True`. A graft that severs the backbone
  silently produces two molecules; that is a validity failure, not an attack.

## 5. Wildcard protection summary

| hazard | guard |
|---|---|
| cut removes an attachment point | never cut `[*]`–heavy bonds |
| graft deletes a `[*]` | `require_attachment_count` from the source |
| graft adds a spurious `[*]` | same check (equality, not `>=`) |
| fragment library accidentally contains `[*]` in the core | reject malformed library entries at load time |

## 6. Edit accounting (the part that must not be fudged)

A fragment swap is **not** one atom edit. For every application report:

* `operator_edit_count` — applications along the path from the original source
  (`Candidate.operator_edits`), which is what the ≤ 3 budget bounds;
* `graph_change_measure` — `graph_metrics.graph_change(...)`, giving
  `element_change_count`, `atom_count_delta`, `bond_count_delta`,
  `component_count_delta` and the derived `structural_change_proxy`.

The two are not interchangeable and must never be presented as one number. A single
application of a two-attachment fragment swap typically moves 2–8 atoms, so a 3-edit
fragment budget is a strictly larger threat model than a 3-edit atom-substitution budget.
Any comparison against phase 12B numbers must state which budget was used.

## 7. Implementation sketch (not implemented)

```python
class FragmentReplacementAttack(AttackOperator):
    def __init__(self, library: Sequence[Fragment], *, protect_attachment=True,
                 max_cut_path_length=4, validator=None): ...

    def edit_cost(self, parent, child) -> int:
        return 1                      # one APPLICATION, not one atom

    def apply(self, candidate) -> list[Candidate]:
        # 1. enumerate eligible cut pairs on the source
        # 2. for each (cut, fragment) build the grafted molecule with RWMol
        # 3. SanitizeMol -> drop on valence failure
        # 4. validator.is_valid (single component, attachment count preserved)
        # 5. reject no-ops, dedupe by canonical identity
        ...
```

## 8. Acceptance criteria before it may be benchmarked

1. Deterministic: same source ⇒ same candidate set.
2. No candidate escapes the §5 guards (tested with adversarial library entries).
3. `graph_change` is computed for every accepted candidate and reported.
4. Runs through the single budgeted scoring path (`AttackEvaluator`); one application is
   one query and the source is scored once, uncharged.
5. A hand-audited sample of ≥ 20 candidates confirms the intended fragment was actually
   swapped (not merely an atom re-typed) — the exact failure mode that made the previous
   operator's name wrong.
6. Only then may the results file be generated, and only with the operator-edit and
   graph-change columns side by side.

## 9. Why this is deferred

The verified benchmark (`docs/FRAMEWORK_V2_ATTACK_RESULTS.md`, `verified_run_1`) must land
first: it establishes the trustworthiness of the pipeline. Adding a new operator with
unfamiliar failure modes before the pipeline is proven would repeat the original mistake
of measuring an implementation artefact and calling it science.
