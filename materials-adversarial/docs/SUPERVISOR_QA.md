# Supervisor Q&A

**What is bandgap?**
The bandgap is the energy difference between the highest occupied electronic state (valence band) and the lowest unoccupied state (conduction band) in a solid material. It dictates whether a polymer behaves as an insulator, semiconductor, or conductor.

**How was the dataset bandgap computed?**
The polyVERSE dataset labels were originally computed using Density Functional Theory (DFT). However, the exact functional, pseudopotential, and periodic boundary conditions used for those specific values were not preserved in the dataset metadata.

**What do we know and not know about that computation?**
We know the numerical target values (in eV) and the 2D graph topologies (SMILES) of the repeat units. We *do not* know the 3D folded geometry of the polymer chains, the vacuum padding, the k-point mesh, or the exact DFT software version used.

**Why use SMILES?**
SMILES is the de facto standard for storing chemical structures as text strings. Sequence models like Transformers ingest text tokens natively, making SMILES the most straightforward way to train large language models on chemistry. 

**Why did Transformer fail on equivalent representations?**
A single molecule can be written as many different, valid SMILES strings depending on which atom you start from. The Transformer treats these as entirely different sequences, failing to inherently learn that the underlying 3D object is identical.

**Why does GraphMPNN fix that?**
The GraphMPNN operates on the actual molecular graph (nodes and edges), which is mathematically invariant to the arbitrary ordering of atoms in a text string. 

**What does zero equivalent-SMILES drift actually mean?**
It means the model is structurally immune to representation-preserving attacks. No matter how an attacker permutes the SMILES string, the model's prediction remains identical up to floating-point numerical precision.

**Why is substitution drift not automatically bad?**
Substituting an atom (e.g., swapping Carbon for Nitrogen) is a chemistry-changing attack. Changing the chemistry *should* change the physical bandgap. A model that predicts a different value is doing what physics demands. It is only "bad" (an adversarial error) if the model predicts a massive change while the true physical property barely moves.

**Why was adversarial fine-tuning unsuccessful?**
When we tried to train the sequence model dynamically to recognize equivalent SMILES, it experienced "post-hoc collapse." It memorized the adversarial examples rather than learning the structural invariance, severely degrading its performance on clean test data.

**What was the Phase 11B evaluation bug?**
Transformer evaluation omitted the padding mask. This corrupted Transformer architecture-comparison metrics (fixed in Phase 11C).

**What was the Phase 12 adaptive search bug?**
Edit distance was not strictly bounded relative to the original source. Stateful search accumulated edits ('edit creep'), producing an artificial maximum drift of `5.961 eV`. We fixed this in Phase 12B by strictly bounding to <=3 edits from the original.

**Why is the 5.961 eV result invalid?**
It was derived from the Phase 12 edit creep bug (where edits accumulated without bound relative to the original source), and does not represent a mathematically constrained <=3 edit adversarial perturbation.

**What does 3.19 eV mean?**
It is the maximum prediction drift we successfully induced using bounded <=3-edit adaptive multi-substitution search under the repaired Phase 12B protocol.

**Why do we need DFT/oracle validation?**
Without recalculating the physical bandgap of the new, adversarially-edited polymer using DFT, we have no idea if the 3.19 eV prediction shift is a model hallucination (error) or an accurate reflection of the new molecule's real physical properties. 

**What is the difference between matched oracle and surrogate oracle?**
A matched oracle perfectly reproduces the source dataset's physics and geometry. Because we lack the original 3D geometries, we built a surrogate oracle: a transparent, reproducible pipeline that makes deterministic geometry choices (capped oligomers) to approximate the physics for calibration purposes.

**What remains to complete the true attack→defend loop?**
We must execute the frozen surrogate QC calibration pilot on an HPC cluster, validate the surrogate's alignment with the dataset, evaluate adversarial source/candidate pairs to find true errors, and then retrain the GraphMPNN using oracle-backed ground truth.

**How can this framework generalize to genomics?**
The modular architecture separates the domain representation from the search and oracle verification. By swapping the graph adapter for sequence tokenizers, the RDKit valence checker for biological constraints, and the DFT oracle for binding-affinity models, the same exact adversarial search engine can evaluate DNA/protein predictors.
