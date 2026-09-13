# Model Architecture and Loss Explainer

This document deeply explains the architectures, self-attention, and loss functions used in the materials-adversarial project.

---

## 1. Ordinary Transformer Baseline

**How it works:**
The ordinary Transformer regressor takes a PSMILES sequence (e.g., `*CC*`), tokenizes it into integer IDs, and converts them into dense embedding vectors. Positional embeddings are added so the model knows the order of tokens. The sequence then passes through 2 Transformer encoder layers. Each layer contains a multi-head self-attention mechanism and a feedforward network. After the encoder layers, we apply masked mean pooling to reduce the sequence of vectors into a single vector (the molecular-sequence representation). Finally, a linear layer regresses this vector into a scalar Bandgap prediction.

**Architecture Diagram:**
```
[PSMILES string] 
      ↓ Tokenizer
[Token IDs]
      ↓ Embedding + Positional Encoding
[Sequence of Vectors]
      ↓ 2x Transformer Encoder Layers (w/ Self-Attention)
[Contextualized Sequence]
      ↓ Masked Mean Pooling
[Single Molecular Vector] (d_model=64)
      ↓ Linear Regressor
[Bandgap Prediction (eV)]
```

### Understanding Self-Attention
Our model uses 4 internal attention heads. Self-attention allows each token in the sequence to look at every other token to build a context-aware representation. For example, a `C` token might learn to pay attention to a nearby `O` token if that combination strongly affects the Bandgap.

**Why Attack Families are Unrelated to Attention Heads:**
The 4 attention heads are standard mathematical operations distributed across the sequence. They are **NOT** assigned to human concepts like "Randomization" or "Substitution". The model learns whatever weights minimize the training loss. Claiming that "Head 1 learned randomization" without experimental proof is anthropomorphizing the model.

---

## 2. Two-Branch Architecture (Phase 3)

In Phase 3, we hypothesized we could force the model to separate robustness concepts into explicit branches.

**How it works:**
We kept the shared Transformer encoder up to the pooled molecular vector. Then, instead of directly predicting the Bandgap, we passed the vector through two parallel linear transformations (branches). 

```
[Single Molecular Vector] 
      ↙        ↘
[Branch A]  [Branch B]
      ↘        ↙
      [Fusion]
         ↓ Linear Regressor
[Bandgap Prediction (eV)]
```

### The Failed Latent Specialization Objective
We applied specific losses to force semantic roles:
* **Branch A (Invariance):** `L_inv = MSE(BranchA(emb_clean), BranchA(emb_rand))`
* **Branch B (Sensitivity):** `L_sens = max(0, margin - ||BranchB(emb_clean) - BranchB(emb_sub)||)`
* **Diversity:** `L_div = CosineSimilarity(BranchA, BranchB)`

**What went wrong:** 
The model took an optimization shortcut (representation collapse). Branch A shrunk its variance drastically so that `BranchA(clean)` and `BranchA(rand)` were trivially close to zero. Branch B maintained high variance to handle the actual prediction task. The branches polarized numerically, but did not specialize semantically into "Randomization" vs "Substitution" experts.

---

## 3. Output-Level Robustness Objective (Phase 4)

In Phase 4, we abandoned internal latent losses and moved the robustness constraints to the final output predictions.

### A. Randomization Consistency
For randomization, the underlying molecule is identical. The true Bandgap is the same. We penalize the model if its prediction changes when the syntax changes.

* **Equation:** `L_consistency = lambda_consistency * MSE(pred_clean, pred_rand)`
* **Effect:** The model is directly forced to output stable predictions for equivalent representations.

### B. Substitution Teacher Consistency
For substitution, the chemistry changes. We cannot use the original Bandgap label because the new molecule has an unknown Bandgap. If we force `pred_sub` to equal `target_clean`, we are lying to the model about chemistry.

Instead, we use **Teacher Distillation**. We take a frozen, unregularized model (the Architecture Control) and ask it to predict the Bandgap of the substitution candidate. We then penalize the training model if it deviates from the teacher's prediction.

* **Equation:** `L_teacher = lambda_teacher * MSE(pred_sub, teacher_pred_sub)`
* **Effect:** This does not teach the model the "true" Bandgap of the new molecule. Instead, it acts as a stability anchor. It prevents the model from predicting wild, erratic values (e.g., 20 eV) when presented with out-of-distribution adversarial edits.

---

## 4. How Inference Works

During inference, the model operates exactly like a standard regressor. It accepts a single PSMILES string and outputs a single Bandgap value. 

The branches (if using the Two-Branch architecture) are still computed, fused, and passed to the regressor. The robustness losses are entirely disabled during inference—they exist only to shape the weights during training.
