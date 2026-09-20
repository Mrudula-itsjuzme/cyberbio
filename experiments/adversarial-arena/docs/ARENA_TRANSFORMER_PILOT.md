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
Attacker adaptation metric $G_A(t) = R(A_{t+1},D_t)-R(A_t,D_t)$ was calculated by evaluating both the current and next attacker policies against the same defender checkpoint $D_t$.

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
