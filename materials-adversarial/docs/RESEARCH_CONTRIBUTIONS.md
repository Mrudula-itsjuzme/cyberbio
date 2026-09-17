# Research Contributions

Based on the empirical findings, scientific audit, and final leakage-free benchmarking, the contributions of this research are hierarchically structured to explicitly separate representation invariance from chemical robustness regularization.

> **"The project deliberately separates representation-preserving robustness, where shared supervision is physically justified, from chemistry-changing robustness, where only sensitivity regularization is evaluated without an external property oracle."**

## 1. Primary Vulnerability: Sequence Serialization Dependence
We demonstrate that chemically identical PSMILES serializations produce substantial Transformer prediction variation. By generating randomized but chemically equivalent SMILES (100% graph and formula identity), we observe a mean absolute prediction drift of **0.5991 eV** (RandomSplit baseline) and **0.8968 eV** (ScaffoldSplit baseline). Because repeated inference on identical inputs produces zero drift, this definitively isolates representation/serialization vulnerability rather than stochastic inference noise.

## 2. Primary Defense: Scientifically Clean Augmentation
We introduce representation-preserving multi-SMILES augmentation as a scientifically clean defense. Because $molecule(x') = molecule(x)$, it is physically justified that $E_g(x') = E_g(x)$, allowing the model to safely inherit exact ground-truth labels. This defense is highly effective:
* **Random Split:** Reduces representation drift by ~45.6% ($0.5991 \rightarrow 0.3257$ eV) while slightly improving clean RMSE ($0.6007 \rightarrow 0.5962$ eV).
* **Scaffold Split:** Reduces representation drift by ~54.5% ($0.8968 \rightarrow 0.4079$ eV) and substantially improves structural out-of-distribution predictive accuracy ($0.6998 \rightarrow 0.6630$ eV).

## 3. Secondary Stress Test: Constrained MCMC Exploration
We designed a constrained Markov Chain Monte Carlo (MCMC) search algorithm to explore chemistry-changing local neighborhoods. Guided by RDKit valence checks, Tanimoto similarity thresholds ($S \ge 0.5$), and attachment-star preservation, this method effectively exposes local model sensitivity, discovering physically plausible structural variants that shift model predictions despite high graph similarity.

## 4. Secondary Defense Experiment: Robustness Regularization
We implemented a label-free MCMC consistency regularization term:
$$L_{\text{cons}}=\mathcal{L}\left(f_\theta(x_{\text{MCMC}}),\operatorname{stopgrad}(f_\theta(x))\right)$$
This avoids false target inheritance by treating the original clean prediction as a detached soft target. The regularizer successfully makes the model less sensitive to the MCMC perturbation family (reducing MCMC drift). **Crucially, because the edited structures alter molecular identity and lack independently computed $E_g$ labels, this demonstrates robustness regularization rather than evidence of improved physical accuracy.**

## 5. Future Definitive Experiment: Physical Oracle Integration
To definitively measure physical accuracy on chemistry-changing candidates, future work must substitute the predictive consistency regularizer with a dedicated physical oracle. By independently calculating $E_g(x_{\text{MCMC}})$ using Density Functional Theory (DFT) or retrieving it from a validated property database, adversarial training can move beyond regularization into true physically supervised robust optimization.
