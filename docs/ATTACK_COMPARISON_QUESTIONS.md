# Attack Comparison Question Bank

## Easy and conceptual

1. What is an adversarial attack in this study?
2. Why is Random a meaningful baseline?
3. What is a target-model query?
4. What is a proposal?
5. Why can proposal count exceed query count?
6. What does a validator do?
7. Why is RDKit parsing not physical validation?
8. What is absolute prediction drift?
9. How is Bio-Cyber success defined?
10. Why is Materials success left `NA`?
11. What does a query budget prevent?
12. Why use the same source instances?
13. Why freeze the canonical project?
14. What is a negative result?
15. Why retain dead ends?
16. What is cross-domain generalization?
17. Why can domain validity rates not be pooled?
18. What is a proposal cap?
19. Why report runtime?
20. When should a method be canonical?

## Mathematical

21. State the general adversarial objective.
22. Define `C(x)`.
23. Write the Metropolis-Hastings acceptance probability.
24. What does `pi(x)` represent here?
25. What symmetry assumption removes the proposal ratio?
26. What temperature appears in the implemented MCMC rule?
27. Define evolutionary fitness.
28. How is occlusion attribution calculated?
29. Write total query accounting for attribution.
30. Define drift per target query.
31. Write the generator cross-entropy objective.
32. Why mask padding in generator loss?
33. Define RL state, action, and transition.
34. Give the proposed shaped reward.
35. Why are RL reward coefficients not “results”?
36. Write the classical GAN minimax objective.
37. Why is token sampling nondifferentiable?
38. What is median paired difference?
39. State the Wilcoxon null hypothesis.
40. Why not default to an unpaired test?

## Implementation and reproducibility

41. Where are frozen source manifests stored?
42. Which seed is in the final recorded comparison?
43. Which target-query budgets were used?
44. How is `MAX_PROPOSALS` related to Q?
45. What three checks follow RDKit parsing?
46. How are attribution queries counted?
47. What is elitism?
48. What stops evolutionary search?
49. What stops MCMC?
50. How are generator calls accounted for?
51. What is the generator architecture?
52. Is the generator a GAN?
53. How is test leakage reduced in generator training?
54. What happened in the trained Materials generator run?
55. Why was the fast untrained script invalid?
56. How was the invalid control removed from final analysis?
57. How do you rebuild final analysis?
58. What happens if `--llm` is requested without setup?
59. Why are mock providers allowed in tests?
60. Why are mock-provider rows forbidden in final evidence?

## Statistical and reviewer-style

61. Why use paired Wilcoxon tests?
62. What is the sample size per domain?
63. Why are p-values exploratory?
64. What multiple-comparison issue exists?
65. Why report effect size with p-value?
66. Why is “statistically better” too strong here?
67. How could seed 42 bias conclusions?
68. Why can attribution-high fail?
69. Why can MCMC fail under tight constraints?
70. Why can evolutionary search fail at budget 5?
71. Why can a trained generator produce invalid SMILES?
72. Why must an LLM output pass RDKit?
73. Why cannot an LLM validate chemistry by itself?
74. What would a real LLM run need to log?
75. What makes RL sample-inefficient?
76. How does reward shaping change RL behavior?
77. What is GAN mode collapse?
78. Why not use RandomSMILES alone?
79. Why not use gradient attacks for this comparison?
80. Why not replace the target with a GNN?
81. Why not put DFT inside every attack step?
82. What physical claim is unsupported by drift alone?
83. Why is target-query efficiency incomplete by itself?
84. What evidence would justify MCMC promotion?
85. What evidence would justify generator promotion?
86. What evidence would unblock LLM conclusions?
87. What evidence would unblock RL conclusions?
88. How should timeout failures be reported?
89. Why is missing evidence `NA` rather than zero?
90. What is the strongest defensible final conclusion?
