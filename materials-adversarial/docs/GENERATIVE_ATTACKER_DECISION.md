# Generative Attacker Decision

**Recommendation:** Do NOT implement GAN/VAE/Diffusion at this stage.

**Rationale:**
- The dataset size is extremely small (~4.2k samples), which is typically insufficient to learn a stable and chemically valid generative space.
- Generative stability may be poor.
- Validity constraints would still need to be enforced post-generation.
- The compute and integration effort is currently better spent on discrete, interpretable bounding methods like the ones in the generic framework.
