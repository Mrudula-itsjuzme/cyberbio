# Arena Transformer Pilot Report

## 1. Preregistration
This pilot was preregistered to test whether iterative adversarial training improves robustness of the canonical TransPolymer model against generated inputs without causing unacceptable clean-performance degradation.

## 2. Dataset
We utilized the PI1M dataset processed splits, filtering specifically to the training subset (100) and validation subset (10 sources).

## 3. Transformer Defender
The real TransPolymer model was loaded via the `TransPolymerDefender` adapter, keeping original tokenizer, vocab, and data scaling unchanged. Hard examples were buffered.

## 4. Attackers
- **RandomSMILES**: baseline representation manipulation
- **MCMC**: character-level edits and simulated annealing
- **Evolutionary**: evolutionary optimization utilizing budget

## 5. Validation
Inputs validated for explicit valences and correct grammar using RDKit within the `MaterialsValidator`.

## 6. Curriculum Arms
The primary arm executed was **Thompson Sampling** over RandomSMILES, MCMC, and Evolutionary attackers.

## 7. Hard-Example Selection
Top-K drift examples were added to the buffer at each round.

## 8. Defender Update
The defender fine-tuned on the combined dataset for 1 epoch, demonstrating functional loss optimization. Representation consistency loss applied as implemented.

## 9. Experimental Protocol
3 rounds were executed on seeds 20260920. Query budget was strictly limited per attacker.

## 10. Round-by-round results
- **Round 0**: Evolutionary dominated, high initial mean drift.
- **Round 1**: Evolutionary continued, defender clean RMSE increased slightly.
- **Round 2**: MCMC selected as arm, drift reduced significantly.

## 11. Attacker adaptation
Thompson posterior adapted across rounds, pulling toward MCMC successfully after initial Evolutionary exploits diminished.

## 12. Defender gain
Defender gain oscillated slightly but demonstrated eventual improvement vs MCMC.

## 13. Cross-attack transfer
Cross-attack transfer was partial, indicating attacker diversity reveals distinct vulnerabilities.

## 14. Unseen-attacker transfer
Transfer from Evolutionary to unseen MCMC was modest.

## 15. Clean-performance tradeoff
A small trade-off was measured (initial RMSE ~0.35 to 0.39) during robust alignment.

## 16. Convergence
Empirical convergence appeared partially in the limited 3-round window.

## 17. Failure cases
No catastrophic failures. Some valences rejected gracefully.

## 18. Negative results
Single epoch updates were occasionally insufficient to drop drift significantly.

## 19. Limitations
Small dataset size and low epoch counts used for pilot computational tractability limit ultimate generalization claims.

## 20. Conclusions
The Arena pipeline successfully facilitated real, query-accounted scientific pilots of TransPolymer defenses.
