import re

def patch_file(filepath, patterns):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        original_content = content
        for pattern, repl in patterns:
            content = re.sub(pattern, repl, content, flags=re.MULTILINE)
            
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Patched {filepath}")
    except FileNotFoundError:
        pass

paper_patterns = [
    # Parameter counts
    (r"Ordinary Transformer \(~12M params\)", r"Ordinary Transformer (85,761 params)"),
    (r"Architecture Control \(~12M params\)", r"Architecture Control (90,049 params)"),
    (r"Mixed-robust Transformer \(~12M params\)", r"Mixed-robust Transformer (90,049 params)"),
    (r"Augmented Transformer \(~12M params\)", r"Augmented Transformer (90,049 params)"),
    (r"GraphMPNN \(~5M params\)", r"GraphMPNN (27,585 params)"),
    # Model desc
    (r"A baseline sequence-to-sequence style encoder", r"A baseline Transformer encoder regression model"),
    # History corrections
    (r"5\. Specialization Negative Result: The two-branch architecture failed to improve robustness without performance collapse\.\n6\. Output-level Experiments: Output-level consistency training and adversarial fine-tuning were attempted\.\n7\. Adversarial Fine-Tuning Collapse: Representation-only fine-tuning resulted in post-hoc collapse, indicating the representation was fundamentally flawed\.", 
     r"5. Specialization Negative Result: The two-branch semantic specialization hypothesis was NOT SUPPORTED. The architecture control itself showed useful robustness/performance changes, but auxiliary losses did not demonstrate the intended semantic decomposition.\n6. Output-level Experiments: Output-level consistency training and adversarial fine-tuning were attempted.\n7. Adversarial Fine-Tuning Collapse: Post-hoc adversarial fine-tuning later produced catastrophic degradation/collapse under tested protocols, a separate finding from the architectural control."),
    # Phase 12 corrections
    (r"9\. Early Adaptive-Search Artifact: Unconstrained edits led to massive structural bloat\.", 
     r"9. Early Adaptive-Search Artifact: Early stateful search was not constrained to <=3 edits relative to the ORIGINAL source, allowing edit creep and invalidating the 5.961 eV maximum drift."),
    (r"15\. Phase 12B Attacks: Repaired bounded search revealed a 3.19 eV drift\.", 
     r"15. Phase 12B Attacks: Phase 12B enforced original-source edit distance <=3, revealing a bounded 3.19 eV drift."),
    # Generalization
    (r"seamlessly scales to DNA \(.+?\) and proteins \(.+?\)\.", r"is designed to support future DNA and protein adapters (NOT IMPLEMENTED, NOT EVALUATED, FUTURE WORK).")
]

patch_file("docs/PAPER_DRAFT.md", paper_patterns)

