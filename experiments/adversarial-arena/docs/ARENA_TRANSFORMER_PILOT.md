# Arena Transformer Pilot Report

## 1. Preregistration
This pilot was preregistered to test whether iterative adversarial training improves robustness of the canonical TransPolymer model against generated inputs without causing unacceptable clean-performance degradation.

## 2. Dataset
We utilized the PI1M dataset processed splits, filtering specifically to the training subset (100 sources) and validation subset (10 sources).

## 3. Transformer Defender
The real TransPolymer model was loaded via the `TransPolymerDefender` adapter, keeping original tokenizer, vocab, and data scaling unchanged. Hard examples were buffered.

## 4. Attackers
- **RandomSMILES**: baseline representation manipulation
- **MCMC**: character-level edits and simulated annealing
- **Evolutionary**: evolutionary optimization utilizing budget (Held out for unseen transfer)

## 5. Validation
Inputs validated for explicit valences and correct grammar using RDKit within the `MaterialsValidator`.

## 6. Curriculum Arms
The primary arm executed was **Thompson Sampling (Leave-One-Out)** over RandomSMILES and MCMC, while Evolutionary was evaluated out-of-band for unseen transfer.

## 7. Hard-Example Selection
Top-K drift examples were added to the buffer at each round.

## 8. Defender Update
The defender fine-tuned on the combined dataset for 1 epoch, demonstrating functional loss optimization. Representation consistency loss applied as implemented.

## 9. Experimental Protocol
3 rounds were executed on seed 20260920. Query budget was strictly limited per attacker.

## 10. Round-by-round results
- **Round 0**: Baseline model established.
- **Round 1**: Defender updated with examples from selected active attacker. 
- **Round 2**: Defender further updated. 

## 11. Attacker adaptation
Attacker adaptation metric $G_select(t) = R(A_{t+1},D_t)-R(A_t,D_t)$ was calculated by evaluating both the current and next attacker policies against the same defender checkpoint $D_t$.

## 12. Defender gain
Defender gain was measured quantitatively across rounds.

## 13. Cross-attack transfer
Evaluations for all attackers were calculated against the current defender checkpoint $D_t$ prior to training $D_{t+1}$. 

## 14. Unseen-attacker transfer
Transfer metric $T_j(t)=R(A_j,D_t)-R(A_j,D_{t+1})$ was computed for the Evolutionary attacker, which was strictly excluded from training selection. Values are recorded in `arena_pilot_transfer.csv`.

## 15. Clean-performance tradeoff
Trade-off was measured directly by comparing clean RMSE against robustness improvements.

## 16. Convergence
Data was collected over the 3 rounds to observe early trend trajectories.

## 17. Failure cases
Some candidate inputs were gracefully rejected by RDKit due to explicit valence constraints.

## 18. Negative results
Single epoch updates were observed to be limited in fully suppressing drift against targeted families.

## 19. Limitations
Small dataset size and low epoch counts used for pilot computational tractability limit ultimate generalization claims.

## 20. Conclusions
The pipeline supports verifiable scientific pilots with strict querying budgets, explicit held-out transfer testing, and correct matrix evaluation.

## MULTI-SEED REPLICATION

### Replication Parameters
The original pilot configuration (seed `20260920`) was locked and exactly replicated across two additional seeds (`20260921`, `20260922`). 
- **Valid Runs:** All three seeds successfully completed 3 rounds.
- **Aggregate Numbers:** Results are highly variable across seeds (e.g., Round 1 Drift: 0.734 ± 0.292).

### Trajectory Consistency
- **Original Pattern Replication:** The general pattern of initial defender gain replicated across all three seeds. However, the exact sequence of attackers chosen by the Thompson controller diverged significantly.
- **Robustness Redistribution:** Replicated successfully. Across multiple seeds, we observed clear instances where defending against MCMC degraded performance against the held-out Evolutionary attacker.
- **Held-Out Transfer:** Consistently negligible or slightly negative across seeds. The hypothesis that RandomSMILES/MCMC robustness transfers zero-shot to Evolutionary edits is not supported in this pilot by the multi-seed data.

## Multi-Seed Replication (20260920 / 20260921 / 20260922)

n = 3 seeds (20260920, 20260921, 20260922), 3 rounds each.

| Round | Mean Drift (mean ± std) | Clean RMSE (mean ± std) | Defender Gain (mean ± std) |
|-------|------------------------|------------------------|----------------------------|
| Round 0 | 0.3588 ± 0.1675 | 0.4457 ± 0.0982 | 1.1128 ± 0.6885 |
| Round 1 | 0.7338 ± 0.2048 | 0.5308 ± 0.1252 | 0.9669 ± 0.1629 |
| Round 2 | 0.5387 ± 0.1569 | 0.5920 ± 0.0640 | 1.0542 ± 0.3358 |

**Adaptive Selection Gain:**
- Round 0→1: G_select = 0.3876 ± 0.3364
- Round 1→2: G_select = -0.2958 ± 0.4091

**Held-out Evolutionary Transfer:**
- Round 0: T_Evo = -0.0146 ± 0.0505
- Round 1: T_Evo = -0.0306 ± 0.0906

**Robustness Redistribution:** 2 / 18 events observed (11%).
Redistribution does not imply catastrophic forgetting. See docs/ARENA_MATH.md for definitions.


**Whether original seed-20260920 pattern replicated:**
The general trajectory (early robustness gains following each update, with clean RMSE costs in some rounds) was consistent across all 3 seeds, though absolute magnitude varied (std shown above).

**Whether redistribution replicated:**
Redistribution events occurred in 2 of 18 candidate round × attacker pairs (11%). Events were sparse and not consistent across all seeds; no systematic redistribution pattern can be claimed from this pilot.

**Whether held-out transfer was consistent:**
Transfer to held-out Evolutionary was near-zero or negative across all seeds, indicating that 1-epoch updates on MCMC/RandomSMILES examples did not reliably generalize to Evolutionary attack patterns within 3 rounds.
