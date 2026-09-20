# VIVA QUESTION BANK

## EASY
**Q1: What is the main objective of this project?**
**Answer:** To evaluate and improve the adversarial robustness of Transformer models trained to predict polymer bandgaps from PSMILES sequences.
**Don't Claim:** That we solved all materials discovery problems or perfectly emulate DFT.

**Q2: What is the target property and its unit?**
**Answer:** Bandgap ($E_g$) measured in electron volts (eV).

**Q3: What are the two types of attacks used?**
**Answer:** Representation-preserving (Random SMILES) and chemistry-changing (MCMC local edits).

**Q4: What is prediction drift?**
**Answer:** The absolute difference in the model's prediction before and after an attack.

**Q5: Why do we use RDKit in the pipeline?**
**Answer:** To ensure that any mutated sequence can still be parsed as a syntactically valid molecular graph with closed rings and valid valences.

## MEDIUM
**Q6: Why does RandomSMILES augmentation improve Scaffold Split RMSE?**
**Answer:** It acts as a data regularizer, forcing the model to learn the underlying molecular topology rather than memorizing specific string syntax, improving out-of-distribution structural generalization.

**Q7: How do you prevent MCMC from generating completely unrelated molecules?**
**Answer:** By applying heuristic plausibility filters: preserving exactly two attachment points, protecting branching/ring tokens, enforcing molecular weight bounds, bounding the edit budget, and enforcing a minimum Tanimoto structural similarity.

**Q8: Does sequence length cause the model to fail?**
**Answer:** We can't strictly prove causation, but our analysis shows absolute length change correlates with prediction drift, consistent with the hypothesis that the model uses sequence length as a proxy shortcut.

**Q9: Why not use a Graph Neural Network (GNN)?**
**Answer:** GNNs natively bypass representation sensitivity, but Transformers scale better with data and are currently state-of-the-art for sequence modeling. Our goal is to analyze and fix the sequence-based models, not replace them.

**Q10: What does the cross-attack generalization matrix show?**
**Answer:** That defending against representation changes does not fix sensitivity to local chemical edits, and vice versa. The vulnerabilities are largely orthogonal.

## HARD
**Q11: How do you know the MCMC structures preserve the original bandgap?**
**Answer:** We don't. Without a Density Functional Theory (DFT) oracle, we cannot guarantee the true physical bandgap remains identical. We are evaluating *model consistency* under tightly constrained local edits, not physical invariance.
**Don't Claim:** That the true physical label of the adversarial molecule is exactly the same as the original.

**Q12: Is RandomSMILES really an adversarial attack?**
**Answer:** In the strict sense of generating worst-case perturbations, yes, if we select the string that maximizes drift. Chemically, it's a test of representation invariance. 

**Q13: Why is Scaffold Split harder than Random Split?**
**Answer:** Scaffold splitting separates molecules by their core structural backbone. The test set contains backbones the model has never seen, testing true structural generalization rather than interpolation.

**Q14: Why should we trust results from one seed?**
**Answer:** Deep learning models are non-convex and can get stuck in local minima. A single seed could be a statistical fluke. *(Update this when multi-seed validation completes to state: "We ran it across 3 independent seeds to prove stability.")*

## REVIEWER-STYLE
**Q15: Why didn't you use LLMs or GANs to generate attacks?**
**Answer:** Generative models add immense complexity and confounding factors (e.g., mode collapse). Simple, targeted token-level edits combined with rigorous MCMC search allow us to directly isolate and measure the model's fundamental structural sensitivity.

**Q16: Your baseline model performs worse than state-of-the-art literature. Doesn't this invalidate the robustness claims?**
**Answer:** The goal is to study the *delta* (prediction drift) under attack and the *relative* improvement of defenses. While pushing SOTA accuracy is valuable, the mechanisms of representation sensitivity and shortcut learning remain relevant regardless of absolute baseline capability.
