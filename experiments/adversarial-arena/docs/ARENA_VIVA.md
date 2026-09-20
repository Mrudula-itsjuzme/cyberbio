# Arena Viva Notes

## Purpose of Models
- Ridge regression was exclusively for fast structural engineering verification (smoke system). It provided no scientific evidence.
- The actual TransPolymer pilot proves the scientific usability of the Arena pipeline.

## Defender Updates
Defender updates are designed to balance pure supervised regression on clean targets with consistency training on robust perturbations. Chemistry-changing edits without labels cannot blindly reuse the old clean property label in supervised modes, hence only consistent regularization or holding out as diagnostic are valid paths.

## Transfer and Unseen Measurement
Unseen-attacker transfer is quantified by comparing the performance difference between checkpoints when evaluated by an attacker family not observed during training. Cross-attack evaluations run concurrently at each checkpoint.

## Scientific Replication
Multi-seed configurations prevent overfitting the analysis to one initialization sequence. Adaptive attacker performance is useful for localized stress-testing but its ultimate success rate doesn't necessarily signal an overall robustness breakdown if evaluated isolated from cross-curriculum baseline metrics.
