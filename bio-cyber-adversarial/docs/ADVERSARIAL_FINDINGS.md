# Bio-Cyber Adversarial: Research Findings

## 1. Adversarial Vulnerability of Synthetic Sequence Models
We established a gradient-based iterative token substitution attack on the baseline 1D CNN. 
**Constraints Enforced:**
- Strict sequence length preservation (50 bp).
- Complete masking of the pathogenic motifs (`ACGTACGT` and `TGCATGCA`) to ensure they were never modified or destroyed by the attack.

**Baseline Results:** 
Against the standard, cleanly-trained CNN, the attack achieved a **99.97% success rate**. This demonstrates that sequence classifiers relying on local motifs are exceptionally vulnerable to adversarial noise implanted in the *background* sequence, causing misclassification even when the true motif remains perfectly intact.

## 2. Inefficacy of Standard Adversarial Training
We implemented an Adversarial Training defense, dynamically generating 2-step gradient substitutions during the training loop and training the network to correctly classify these adversarial examples.

**Defended Model Results:**
When evaluated against the 10-step iterative attack, the defended model yielded a **95.63% success rate**. 
While this is a slight reduction from 99.97%, the defense is largely ineffective. 

**Scientific Implication:**
This is a critical finding for the branch. It suggests that standard adversarial training (PGD-style) is insufficient for 1D CNN architectures using max-pooling over discrete token sequences. The max-pooling operation allows the network to be highly sensitive to spurious, non-contiguous features scattered throughout the background sequence that accidentally trigger the motif-detection filters, overpowering the true motif. 

This sets up a compelling foundation for future work on structural robustness, positional encoding, or transformer-based global attention for bio-cybersecurity tasks!
