# Viva Defense Pack

## A. 60-Second Project Explanation
"We investigated the adversarial robustness of machine learning models predicting polymer bandgaps. We found that standard sequence models hallucinate wildly (0.614 eV) just by rewriting the same molecule's SMILES string differently. We fixed this representation flaw by upgrading to a GraphMPNN, reducing that drift to zero. However, when we applied valid, bounded chemical mutations (substitutions/deletions), the GraphMPNN prediction still shifted by up to 3.19 eV. Because these edits physically change the molecule, we cannot definitively call this 'model error' without checking a physical oracle like DFT. We built the calibration pipeline for that oracle, bringing the project right to the boundary of true physical verification."

## B. 3-Minute Project Explanation
"Material informatics relies heavily on deep learning to bypass expensive DFT calculations. However, we discovered that these models are easily fooled. First, we tackled 'representation attacks'—if you feed a sequence model equivalent SMILES strings for the exact same molecule, it gives drastically different bandgap predictions. We attempted to train the sequence model out of this, but it collapsed. The structural fix was abandoning sequences for Graph Neural Networks (GraphMPNN), which inherently understand that the molecule hasn't changed.

Second, we tackled 'chemistry-changing attacks'. We built an adaptive search algorithm that applies strict, chemically valid single-atom substitutions and deletions. We managed to drive the GraphMPNN prediction off by up to 3.19 eV. But here lies the core scientific problem: if I mutate a carbon to a nitrogen, the real physical bandgap *should* change. How do I know if my model's 3.19 eV shift is a hallucination or an accurate physics prediction?

To answer this, we need an independent physical oracle—Quantum ESPRESSO. Because the original dataset lacked 3D geometries, we built a transparent surrogate structure protocol to generate verifiable structures for the oracle. We have prepared a full HPC calibration pilot, freezing the project precisely at the point where computation hands over to external physical verification."

## C. 10-Minute Detailed Explanation
(Expands on the 3-minute explanation by stepping through the project phases in order. Refer to `docs/PROJECT_STORY.md` and `docs/PAPER_DRAFT.md` for the narrative structure. Key points to hit: 1. The sequence baseline, 2. The failure of the two-branch semantic hypothesis, 3. The GraphMPNN structural victory, 4. The Phase 11B padding-mask bug and the unrelated Phase 12 edit creep bug (which caused the fake 5.961 eV result), 5. The necessity of the oracle, 6. The `CALIBRATABLE_SURROGATE` geometry protocol, and 7. The exact frozen state.)

## D. Likely Hard Questions
1. "Why did you use GraphMPNN instead of a more modern Transformer like Graphormer?"
2. "If you don't know the true physical bandgap of your adversarial examples, how can you claim you found an adversarial attack at all?"
3. "Why are insertion attacks fundamentally different from substitutions?"
4. "Your surrogate oracle uses finite oligomers for a 1D periodic dataset. Isn't that physically invalid?"

## E. Concise Answers
1. *Graphormer vs GraphMPNN:* The goal was to prove the necessity of structural permutation invariance, not to chase state-of-the-art architectures. GraphMPNN provides an exact mathematical guarantee of invariance, which perfectly isolated the representation vulnerability.
2. *Adversarial Claim:* I am explicitly *not* claiming I found a proven physical adversarial error. I am claiming I found severe *prediction drift* under bounded chemical edits, and I am proving that you *must* have an oracle to evaluate it. The inability to distinguish drift from error without an oracle is the core contribution.
3. *Insertions vs Substitutions:* Unbounded insertions allow the molecule to grow infinitely. However, the 5.961 eV fake drift was actually caused by the Phase 12 edit creep bug (where states were not bounded to <=3 edits from the original source). Substitutions and deletions preserve the approximate scale of the original scaffold, keeping the search physically bounded.
4. *Oligomer Surrogate:* Yes, finite oligomers differ from infinite periodic chains. That is why it is explicitly designated a `CALIBRATABLE_SURROGATE`. The prepared 9-job HPC pilot is designed precisely to measure the correlation between the finite approximation and the periodic dataset labels before relying on it for adversarial evaluation.

## F. Equations to Know
*   **Target Property (Bandgap):** $E_g = E_{LUMO} - E_{HOMO}$ (in a finite approximation) or the difference between the conduction and valence bands.
*   **Clean MAE:** $\frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |$
*   **Representation Drift:** $|\hat{y}(SMILES_1) - \hat{y}(SMILES_2)|$ for the same molecular graph.

## G. Model Architecture Questions
*Q: How does GraphMPNN achieve invariance?*
A: It updates node states based on the unordered set of local neighbors (message passing) and aggregates the final graph state using a symmetric pooling function (e.g., sum or mean). The order of nodes in memory does not affect the output.

## H. Attack Questions
*Q: How do you guarantee the edits are chemically valid?*
A: We use RDKit's built-in valency models (e.g., `Chem.SanitizeMol`) to strictly reject any generated graph that violates standard chemical bonding rules.

## I. Bandgap/DFT Questions
*Q: What DFT functional did the original dataset use?*
A: The dataset provenance is unresolved regarding the exact functional and basis set. This lack of transparency is exactly why we had to build a standardized, reproducible surrogate protocol with documented choices (e.g., PBE, SSSP efficiency).

## J. Limitations Questions
*Q: Is your test set compromised?*
A: Yes, it is an "exposed" test set. We repeatedly evaluated against it during the iterative debugging of our adversarial framework. The metrics should be viewed as diagnostic rather than pristine, generalizing bounds.

## K. "What is novel here?" Answer
"The explicit decoupling of representation-preserving drift from chemistry-changing drift, and the formalization of the physical oracle validation loop. While adversarial ML is common, applying it to physics requires separating syntax flaws (which ML models can fix) from physical reality (which requires independent verification)."

## L. "What failed?" Answer
"Our initial attempt to solve the sequence vulnerability by training a specialized 'semantic' two-branch Transformer collapsed entirely. We also had a major bug where edit creep (accumulating edits beyond the original budget) caused a fake 5.961 eV 'attack' drift. These failures were critical because they proved that dataset augmentation cannot fix structural flaws, and that physical bounds are mandatory in chemical adversarial search."

## M. "What would you do next?" Answer
"Execute the prepared 9-job surrogate Quantum ESPRESSO pilot on a SLURM cluster to calibrate the geometry. If it correlates with the dataset, I would immediately test the top 5 adversarial source/candidate pairs against the oracle to find our first mathematically proven physical adversarial error."

### Addendum: General Framework V2
"We expanded the chemistry-specific adversarial search into a fully generic structured-domain framework (Framework V2). This generalization explicitly defines interfaces for representation adaptation, validity constraints, and search objectives. We implemented new attack families—Motif Replacement and Scaffold-Preserving edits—along with evolutionary search and LLM-guided proposals. This demonstrates that the core separation of representation-preserving from chemistry-changing stress applies universally across structured material and biological domains."
