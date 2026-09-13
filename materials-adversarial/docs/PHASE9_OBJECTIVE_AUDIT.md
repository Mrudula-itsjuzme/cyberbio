# Phase 9 Objective Audit

## 1. Originally Proposed Objective
The original Phase 9 plan proposed a closed-loop adversarial training experiment using **chemistry-changing** adaptive attacks (multi-edit substitutions). For these adversarial candidates ($x_{adv}$), the objective function proposed for training the new defender ($D1$) was:
$$ \text{target}(x_{adv}) = D0(x_{adv}) $$
This was intended as a self-distillation or adversarial consistency objective, forcing $D1$ to match the prediction of the frozen $D0$ model on the adversarial candidate.

## 2. Why it is Circular
This objective is circular because the adversarial candidate $x_{adv}$ was explicitly discovered by an attacker trying to maximize the prediction drift from the original representation:
$$ J(x_{adv}) = |D0(x_{adv}) - D0(x_{original})| $$
By defining the training target as $D0(x_{adv})$, we are taking the model's *already attacked, erroneous output* and using it as ground truth.

## 3. Why D0(x_adv) Distillation Does Not Defend Prediction Drift
Training $D1$ to reproduce $D0(x_{adv})$ does not teach the model to resist adversarial drift or pull the prediction back toward a stable, correct value. Instead, it teaches $D1$ to **imitate $D0$'s failure mode**. 
If $D0(x_{original}) = 4$ eV and the attacker successfully forces $D0(x_{adv}) = 6$ eV, setting the target for $x_{adv}$ to 6 eV simply reinforces the vulnerability rather than correcting it.

## 4. Why Original-Label Inheritance is Invalid
One might attempt to fix this by assigning the measured Bandgap of $x_{original}$ to $x_{adv}$:
$$ \text{target}(x_{adv}) = y_{measured} $$
However, because the substitution attacks actually change the molecular identity of the polymer (e.g., swapping a carbon for an oxygen), the true physical Bandgap of $x_{adv}$ is unknown and likely differs from $x_{original}$. **Chemistry-changing adversarial candidates are not label-preserving.** Enforcing the original label would force the model to learn incorrect physical chemistry.

## 5. Attacks Permitting True Consistency Training
True adversarial training consistency is scientifically valid only when the adversarial transformation does not change the physical entity or property being predicted. In our domain, this corresponds to a **Representation Adversary**:
An attacker that searches over canonically equivalent molecular-string representations (e.g., randomized SMILES). Here, $molecule(x_{adv}) == molecule(x_{original})$, so the measured Bandgap label is legitimately shared, and consistency constraints ($D(x_{adv}) \approx D(x_{original}) \approx y_{measured}$) are physically sound.

## 6. Attacks Requiring Independent Property Supervision
**Chemistry-Changing Adversaries** (substitution, deletion, insertion) alter molecular identity. For these candidates, the true Bandgap is unknown. These attacks are excellent for stress-testing, sensitivity analysis, and measuring model failure boundaries, but they **cannot** be used for direct supervised adversarial correction without an **independent property oracle**. 
To train on chemistry-changing candidates, one would require an independent target source (e.g., DFT calculations, real-world measurement, or a separately justified surrogate teacher) to provide the true target $T(x_{adv})$.

## Redefined Closed-Loop Problem
The corrected Phase 9 experiment will focus entirely on a **Representation-Preserving Adaptive Attacker**. The attacker will search over canonically equivalent molecular strings to maximize prediction drift, and the defender will be trained to enforce invariance and correctly predict the shared physical label under these worst-case equivalent representations.
