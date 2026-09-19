# FINAL VIVA SUMMARY
- **Problem**: Sequence models for physical/biological representations are vulnerable to syntax-preserving and chemistry-changing perturbations.
- **Dataset**: PI1M polymers and 20,000 synthetic DNA sequences.
- **Main Vulnerability**: High representation dependence, lacking physical grounding.
- **Main Defense**: Adversarial training and consistency regularization.
- **Why Random-SMILES matters**: Isolates representation drift from chemical drift.
- **Why chemistry-changing attacks are different**: Alter ground-truth properties.
- **What LLM/RL/Gen mean**: Highly adaptive token-level search.
- **Why physical oracle validation is necessary**: Without DFT, chemical validity is only syntactic.
- **Limitations**: No physical oracle executed, LLM blocked.
