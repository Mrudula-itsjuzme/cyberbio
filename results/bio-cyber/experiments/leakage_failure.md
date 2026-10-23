# Bio-Cyber Dataset Leakage Failure

The bio-cyber dataset fails the data leakage audit. 

Simple linear classification over k-mer counts achieved:
- 1-mer accuracy: 50.9%
- 2-mer accuracy: 93.6%
- 3-mer accuracy: 98.9%

Because a trivial linear model on 3-mers can perfectly classify the dataset (98.9% accuracy), the task is compromised by shortcut features. The dataset cannot support robust deep learning claims. Bio-Cyber experiments (Phases I-M) have been halted to prevent publishing results on a compromised dataset.
