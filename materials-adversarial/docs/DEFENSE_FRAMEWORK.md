# Closed-Loop Adversarial Defender Framework

## 1. Min-Max Adversarial Optimization Objective

The defender objective unifies model training and adversarial attack generation into a single **min-max optimization framework**:

$$\min_{\theta} \mathcal{L}_{\text{defender}}(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{clean}}\left(f_\theta(x), y\right) + \lambda \mathcal{L}_{\text{cons}}\left(f_\theta(x_{\text{adv}}'), \operatorname{stopgrad}(f_\theta(x))\right) \right]$$

where:
- $\theta$ represents the trainable weights of the sequence Transformer model.
- $\mathcal{L}_{\text{clean}}(f_\theta(x), y) = \left(f_\theta(x) - y\right)^2$ is Mean Squared Error (MSE) on unperturbed training samples.
- $x_{\text{adv}}' = \text{Attacker}(x; f_\theta)$ is the adversarial candidate generated dynamically by the probabilistic MCMC attack generator acting on input $x$.
- $\mathcal{L}_{\text{cons}}(f_\theta(x_{\text{adv}}'), \operatorname{stopgrad}(f_\theta(x))) = \left(f_\theta(x_{\text{adv}}') - \operatorname{stopgrad}(f_\theta(x))\right)^2$ is the label-free consistency regularization loss, preventing false target inheritance.
- $\lambda$ is the regularization weight parameter (default $\lambda = 0.5$).

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           CLOSED-LOOP DEFENDER TRAINING ITERATION                       │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Sample Mini-Batch: Clean polymer representations {(x_i, y_i)}_{i=1}^B                │
│ 2. Forward Pass (Clean): Compute predictions f_theta(x_i) and L_clean = MSE(f_theta(x),y) │
│ 3. Attacker Generation Step:                                                            │
│    - Run ProbabilisticMCMCAttack(x_i; f_theta) for N_steps=10                           │
│    - Validate candidates via ChemicalPlausibilityValidator (S_Tanimoto >= 0.5)           │
│    - Produce adversarial mini-batch {x'_adv,i}_{i=1}^B                                  │
│ 4. Forward Pass (Adversarial): Compute predictions f_theta(x'_adv,i)                    │
│    - Compute L_cons = MSE(f_theta(x'_adv), stopgrad(f_theta(x)))                        │
│ 5. Loss Fusion & Backprop:                                                              │
│    - Total Loss L = L_clean + lambda * L_cons                                           │
│    - Update model weights theta via Adam optimizer                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dynamic Online Adversarial Candidate Generation

Unlike static data augmentation that appends pre-computed perturbed sequences to a fixed training set, our closed-loop defender generates **dynamic online adversarial candidates** during training:
1. In each training mini-batch, the attacker queries the current model state $f_\theta$.
2. The attacker executes Metropolis-Hastings MCMC search to identify current model vulnerabilities.
3. The defender computes loss on these newly discovered vulnerabilities and updates parameters $\theta$ via backpropagation.

This dynamic feedback loop ensures that as the defender model learns to resist specific mutations, the attacker continuously evolves to search for new fragile directions.

---

## 3. Loss Weight Optimization ($\lambda$) and Tradeoff Analysis

The parameter $\lambda \in [0.0, 1.0]$ controls the balance between clean accuracy and adversarial robustness:

- **$\lambda = 0.0$ (Clean Baseline Training)**: Optimizes purely for clean MSE. Yields lowest clean RMSE ($1.0995\text{ eV}$), but leaves the model highly vulnerable to adversarial drift ($0.1072\text{ eV}$).
- **$\lambda = 0.5$ (Balanced Defense - Default)**: Optimizes clean and adversarial loss equally. Achieves a **20.63% reduction in prediction drift** ($0.0766\text{ eV}$) while preserving acceptable clean accuracy ($1.3461\text{ eV}$ clean RMSE).
- **$\lambda = 1.0$ (Pure Adversarial Training)**: Forces model parameters to focus exclusively on adversarial inputs, causing clean accuracy degradation without further robustness gains (diminishing returns).

---

## 4. Optimization Hyper-Parameters

- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$).
- **Learning Rate**: $\eta = 1\times 10^{-3}$.
- **Batch Size**: $B = 16$.
- **Adversarial Steps per Batch**: $N_{\text{steps}} = 10$ MCMC steps per sample.
- **Training Epochs**: 5 epochs per seed.
