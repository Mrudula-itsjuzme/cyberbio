V2 is acceptable for deep-sequence experiments only if:
- no split duplicates exist across train, validation, and test splits
- classes are exactly balanced in all splits
- sequence lengths are exactly matched or the sequence-length baseline remains at chance performance
- low-order compositional baselines (1-mer through 6-mer) do not nearly solve the task (should be substantially far from 1.0)
- best shallow sequence baseline (incorporating motifs and local counts) remains substantially below the intended sequence model performance
