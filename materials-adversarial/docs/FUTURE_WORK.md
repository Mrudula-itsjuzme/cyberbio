# Future Research Directions

Future work is categorized into near-term engineering extensions and long-term research directions:

## 1. Near-Term Engineering Extensions

1. **Cross-Architecture Attack Transferability**: Evaluate whether adversarial candidates generated against the sequence Transformer successfully attack distinct model families, such as Graph Message Passing Neural Networks (`GraphMPNNPredictor`) or XGBoost descriptor baselines.
2. **Learned Generative Proposal Distributions**: Replace static bioisosteric substitution matrices with a learned conditional VAE or RL policy network $q_\phi(x' | x)$ to propose subtle, high-impact chemical modifications.
3. **Synthetic Accessibility (SA) Filtering**: Integrate RDKit SA-Score and SAScore filters directly into `ChemicalPlausibilityValidator` to enforce wet-lab synthetic feasibility bounds ($\text{SA} \le 4.0$).
4. **Multi-Property Joint Adversarial Learning**: Extend closed-loop adversarial training to multi-task Transformer models predicting bandgap ($E_g$), glass transition temperature ($T_g$), and electron affinity simultaneously.

---

## 2. Long-Term Research Directions

1. **Cyberbiosecurity & Genomic Sequence Hardening**: Adapt the sequence-level adversarial attack and defense framework to genomic sequence classifiers, screening algorithms for synthetic DNA orders, and biosecurity threat detection pipelines.
2. **Dynamic Honeypot Attacker-Defender Systems**: Construct game-theoretic honeypot environments where defenders deploy deceptive surrogate outputs to isolate and identify malicious automated querying agents.
3. **RAG & Retrieval Database Security**: Extend adversarial vulnerability research to Retrieval-Augmented Generation (RAG) pipelines in scientific LLMs, testing resistance to scientific literature database poisoning.
4. **Generative Adversarial Networks (GANs) for Materials Discovery**: Incorporate robust property surrogates into GAN discriminator loops to prevent generative models from producing fragile or adversarial material structures.
