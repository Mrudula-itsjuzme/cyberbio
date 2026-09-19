import os

with open('README.md', 'r') as f:
    content = f.read()

# Fix 1: intro paragraph
content = content.replace("closed-loop min-max defender training", "label-free consistency regularization for chemistry-changing stress tests")
content = content.replace("Closed-loop min-max defender training", "Label-free consistency regularization for chemistry-changing stress tests")

# Fix 2: documentation table
content = content.replace("Closed-loop min-max adversarial training", "Representation-preserving augmentation and label-free MCMC consistency regularization")

with open('README.md', 'w') as f:
    f.write(content)

