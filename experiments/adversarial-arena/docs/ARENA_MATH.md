# Arena Mathematical Definitions

## Notation
- `D_t` — defender checkpoint after round t (D_0 = initial pretrained baseline)
- `R(A, D)` — mean robustness drift of attacker A against defender D; higher = attacker causes more prediction shift
- `A_t` — attacker family selected by the Thompson controller in round t

---

## 1. Defender Gain

```
Defender Gain(t) = R(A_t, D_{t-1}) / R(A_t, D_t)
```

> Ratio > 1 means the defender became more robust to the active attacker after training.
> Ratio < 1 means the defender's vulnerability increased.

---

## 2. Adaptive Selection Gain

```
G_select(t) = R(A_{t+1}, D_t) - R(A_t, D_t)
```

- Both terms evaluated against the **same defender checkpoint** D_t.
- A_t and A_{t+1} are the attacker families selected by the controller in consecutive rounds.
- **Positive**: the next-selected family is stronger against the same defender.
- **Negative**: the next-selected family is weaker.

> This measures curriculum selection quality, NOT within-attacker policy adaptation.
> Do NOT conflate with Attacker Adaptation (reserved for learned internal policy updates: CEM, SMC, RL, fine-tuned generators).

---

## 3. Held-Out Transfer

```
T_j(t) = R(A_j, D_t) - R(A_j, D_{t+1})
```

- A_j = Evolutionary (held out from Thompson training).
- D_{t+1} trained using hard examples from {RandomSMILES, MCMC} only.
- **Positive**: defender became less vulnerable to the held-out attack family.
- **Negative**: defender became more vulnerable.
- **Near zero**: negligible transfer.

---

## 4. Robustness Redistribution

Redistribution is observed in round t when ALL of:

```
active_gain  = R(A_t, D_{t-1}) - R(A_t, D_t)   > 0   (active attacker suppressed)
other_gain   = R(A_j, D_{t-1}) - R(A_j, D_t)   < 0   (other attacker worsened)
```

> This is a directional empirical observation. It does NOT imply catastrophic forgetting
> without additional mechanistic evidence.

---

## 5. Clean-Performance Tradeoff Classification

For each round t → t+1:
```
Delta_clean(t)   = RMSE_clean(D_{t+1}) - RMSE_clean(D_t)
Delta_robust(t)  = R(A, D_t) - R(A, D_{t+1})   [positive = better]
```

Classification:
- `ROBUST_WIN_CLEAN_WIN`  — Δ_robust > 0 AND Δ_clean ≤ 0
- `ROBUST_WIN_CLEAN_COST` — Δ_robust > 0 AND Δ_clean > 0
- `ROBUST_LOSS_CLEAN_WIN` — Δ_robust ≤ 0 AND Δ_clean ≤ 0
- `ROBUST_LOSS_CLEAN_COST`— Δ_robust ≤ 0 AND Δ_clean > 0

---

## 6. Thompson Reward

```
reward(t) = min(1.0, mean_drift / (mean_drift + 1))   clipped to [0,1]
```

Used to update Beta(α, β) posterior for the active attacker arm.
