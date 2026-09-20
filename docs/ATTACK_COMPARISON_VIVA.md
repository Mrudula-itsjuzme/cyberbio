# Attack Comparison Viva Guide

An adversarial attack searches for an allowed input `x'` that maximizes model-output change `|f(x')-f(x)|`. The validator defines “allowed”; the query budget defines how often the target model may be evaluated. A valid-looking string is not automatically physically meaningful.

**Random search** samples edits independently. We tried it because every complex method needs a simple baseline. It is cheap and unbiased by model explanations, but wastes proposals in constrained spaces.

**MCMC** is a stateful local walk. A proposal `x'` from `q(x'|x)` is accepted with `min(1,pi(x')q(x|x')/[pi(x)q(x'|x)])`. Here the simplified symmetric form uses `pi(x) proportional exp(drift/10)`. We tried it to keep promising local changes; it can mix slowly.

**Evolutionary search** maintains candidates, mutates them, selects by drift fitness, and retains elites. We tried it for broader exploration than one MCMC chain. It costs many proposals and small budgets undermine population dynamics.

**Attribution attacks** occlude each token and measure output change. High targets the largest score; low and random are controls. We tried them to test whether sensitivity guides edits. Attribution itself costs one target query per token and is not necessarily causal.

**Generative proposals** learn sequence regularities. The GRU denoising autoencoder minimizes masked token cross entropy and samples proposals, which still pass the common validator. We tried trained versus untrained models to isolate learning. Bio runs exist; Materials trained accepted none and Materials untrained is incomplete.

**LLM proposals** would use an external model only to suggest candidates. We tried the software path because language models may encode structural priors, but the validator remains authoritative. Mock tests are not efficacy evidence; real execution is credential-blocked.

**RL** frames editing as state, action, transition, and reward. It might learn adaptive multi-step policies, but sparse validity/drift rewards and target-query demand make it sample-inefficient. It was not run.

**GANs** train a generator against a discriminator. Discrete tokens block ordinary gradients, and mode collapse/invalid strings are serious. No GAN was implemented; the denoising autoencoder must not be called one.

Fair comparison holds sources, model, validator, seed protocol, target-query budget, and metrics fixed. It separately reports proposals, attribution queries, generator inference, LLM tokens/calls, runtime, and successes. Paired Wilcoxon tests use the same sources because independent tests would discard the experimental pairing. Cross-domain generalization means an advantage survives both domains under their own valid semantics; it does not mean pooling DNA and polymer metrics.

The promotion rule is deliberately strict: valid evidence, reproducibility, fair budgets, meaningful advantage, and acceptable cost. No new family currently passes all five.
