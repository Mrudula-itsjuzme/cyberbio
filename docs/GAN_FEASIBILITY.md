# GAN Feasibility

Status: **REJECTED_METHOD** for this branch. No GAN was implemented, approximated, or measured. The GRU denoising autoencoder is not a GAN.

Ordinary GAN gradients do not pass through discrete token sampling. Straight-through or policy-gradient estimators add bias/variance; adversarial sequence training is unstable; mode collapse can reduce proposal diversity; and unconstrained generators readily produce invalid DNA/SMILES. Chemical validators also create sparse discontinuous feedback. A GAN-like method would be reconsidered only with a precise discrete estimator, validator-aware decoding, diversity metrics, matched compute/query accounting, and trained/untrained controls. It is not suitable as the current canonical approach.
