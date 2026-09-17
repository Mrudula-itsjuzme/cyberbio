import os
import glob

def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(filepath, 'w') as f:
            f.write(content)

# PROJECT_STORY.md
replace_in_file("docs/PROJECT_STORY.md",
    "*(Note: An earlier evaluator bug falsely reported 5.961 eV due to unconstrained structure inflation; this bug was caught, corrected, and the 5.961 eV result was marked invalid).* ",
    "*(Note: An earlier search bug in Phase 12 falsely reported 5.961 eV due to edit creep (states not strictly constrained to <=3 edits from the original source); this bug was caught in Phase 12B, corrected, and the 5.961 eV result was marked invalid).* "
)
replace_in_file("docs/PROJECT_STORY.md",
    "Our adaptive adversarial search uncovered valid, rule-constrained chemical edits that induced prediction changes up to `≈ 3.19 eV`.",
    "The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately `3.19 eV`. (Deletion was evaluated separately as a fixed stress)."
)

# PAPER_DRAFT.md
replace_in_file("docs/PAPER_DRAFT.md",
    "*   **Phase 11B Evaluator Bug:** An early evaluation bug permitted unbounded padding/insertion, leading to artificial geometric bloat and an invalid 5.961 eV drift. This confirmed the necessity of strict, bounded edit constraints.",
    "*   **Phase 11B Evaluator Bug:** Transformer evaluation omitted the padding mask. This corrupted Transformer architecture-comparison metrics (fixed in Phase 11C).\n*   **Phase 12 Adaptive-Search Bug:** Edit distance was not strictly bounded relative to the original source, leading to edit creep and an invalid 5.961 eV drift (fixed in Phase 12B to enforce <=3 edits)."
)
replace_in_file("docs/PAPER_DRAFT.md",
    "*(Note: Insertion operations were not canonically tested to prevent unbounded structure inflation, which previously caused a bug reporting 5.961 eV drift.)*",
    "*(Note: Insertion operations were not canonically tested. The 5.961 eV drift was due to a separate Phase 12 edit creep bug, not the Phase 11B padding bug.)*"
)
replace_in_file("docs/PAPER_DRAFT.md",
    "uncovered valid, rule-constrained chemical edits that induced prediction changes up to a repaired maximum bound of ≈ 3.19 eV.",
    "produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV using bounded multi-substitutions."
)
replace_in_file("docs/PAPER_DRAFT.md",
    "causing large prediction changes (up to 3.19 eV).",
    "The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV."
)
replace_in_file("docs/PAPER_DRAFT.md",
    "substitutions and deletions). The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV",
    "substitutions and deletions). The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV."
)


# PRESENTATION_SUMMARY.md
replace_in_file("docs/PRESENTATION_SUMMARY.md",
    "We developed a mathematically bounded adaptive search framework that finds chemically valid edits (substitutions/deletions) causing large prediction changes (up to `3.19 eV`).",
    "The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately `3.19 eV`. Deletion was evaluated separately as a fixed stress."
)
replace_in_file("docs/PRESENTATION_SUMMARY.md",
    "* **Evaluator Inflation**: An early evaluation bug permitted unbounded geometric bloat, falsely registering a `5.961 eV` drift. This was caught by implementing strict graph edit constraints, bounding the true maximum drift to `3.19 eV`.",
    "* **Phase 11B Evaluator Bug**: Transformer evaluation omitted the padding mask, corrupting metrics.\n* **Phase 12 Adaptive Search Bug**: Edit creep occurred because search states were not strictly bounded to <=3 edits relative to the original source, falsely registering a `5.961 eV` drift. This was corrected in Phase 12B."
)

# SUPERVISOR_QA.md
replace_in_file("docs/SUPERVISOR_QA.md",
    "**What was the Phase 11B evaluation bug?**\nThe attacker was permitted to infinitely inflate the molecule (unbounded insertion), adding arbitrary atoms to maximize the loss. This resulted in chemically valid but meaningless structures, producing an artificial maximum drift of `5.961 eV`. We fixed this by strictly bounding the edit operators.",
    "**What was the Phase 11B evaluation bug?**\nTransformer evaluation omitted the padding mask. This corrupted Transformer architecture-comparison metrics (fixed in Phase 11C).\n\n**What was the Phase 12 adaptive search bug?**\nEdit distance was not strictly bounded relative to the original source. Stateful search accumulated edits ('edit creep'), producing an artificial maximum drift of `5.961 eV`. We fixed this in Phase 12B by strictly bounding to <=3 edits from the original."
)
replace_in_file("docs/SUPERVISOR_QA.md",
    "It was derived from the aforementioned unbounded evaluator bug and does not represent a mathematically constrained adversarial perturbation.",
    "It was derived from the Phase 12 edit creep bug (where edits accumulated without bound relative to the original source), and does not represent a mathematically constrained <=3 edit adversarial perturbation."
)
replace_in_file("docs/SUPERVISOR_QA.md",
    "we successfully induced using strictly bounded, chemically valid edits (substitution/deletion only) under our adaptive search framework.",
    "we successfully induced using bounded <=3-edit adaptive multi-substitution search under the repaired Phase 12B protocol."
)

# VIVA_DEFENSE_PACK.md
replace_in_file("docs/VIVA_DEFENSE_PACK.md",
    "We also had a major bug where unconstrained graph insertions caused a fake 5.961 eV 'attack' due to structural bloat.",
    "We also had a major bug where edit creep (accumulating edits beyond the original budget) caused a fake 5.961 eV 'attack' drift."
)
replace_in_file("docs/VIVA_DEFENSE_PACK.md",
    "3. The GraphMPNN structural victory, 4. The bounded chemistry stress and the 5.961 eV inflation bug",
    "3. The GraphMPNN structural victory, 4. The Phase 11B padding-mask bug and the unrelated Phase 12 edit creep bug (which caused the fake 5.961 eV result)"
)
replace_in_file("docs/VIVA_DEFENSE_PACK.md",
    "Unbounded insertions allow the molecule to grow infinitely, inflating the property prediction artificially (our Phase 11B bug).",
    "Unbounded insertions allow the molecule to grow infinitely. However, the 5.961 eV fake drift was actually caused by the Phase 12 edit creep bug (where states were not bounded to <=3 edits from the original source)."
)

# PROJECT_FROM_SCRATCH.md
replace_in_file("docs/PROJECT_FROM_SCRATCH.md",
    "We built an automated search algorithm to test thousands of small, valid chemical tweaks, looking for the one that made the GNN's prediction change the most. We found a small tweak that shifted the prediction by a massive 3.19 eV!",
    "We built an automated search algorithm to test thousands of small, valid chemical tweaks. The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV!"
)

# PAPER_FIGURE_PLAN.md
replace_in_file("docs/PAPER_FIGURE_PLAN.md",
    "bugs (inflation bug)",
    "bugs (Phase 11B padding bug and Phase 12 edit creep bug)"
)

# NOVELTY_POSITIONING.md
replace_in_file("docs/NOVELTY_POSITIONING.md",
    "We expose the danger of unbounded structural operations (insertions causing geometric bloat) in adversarial chemical search, which can lead to artificially inflated vulnerability metrics (the 5.961 eV invalidation).",
    "We expose the danger of edit creep (stateful search accumulating edits beyond the budget from the original source) in adversarial chemical search, which led to artificially inflated vulnerability metrics (the 5.961 eV invalidation in Phase 12)."
)

