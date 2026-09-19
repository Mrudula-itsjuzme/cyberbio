# GENERATIVE ATTACK FEASIBILITY

## Analysis of Methods
- **GAN**: Discrete token difficulty, mode collapse likelihood.
- **VAE**: Better for continuous representations, but complex mapping to discrete.
- **Autoregressive sequence generator**: Most defensible for sequence native generation and source conditioning.
- **Discrete diffusion**: Promising but high inference overhead.

## Conclusion
An autoregressive sequence generator is selected as the most defensible method for exploring sequence modifications natively.
