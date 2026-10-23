# Polymer Quantum Representation Strategy
To evaluate a polymer sequence `[*]...[*]` via DFT:
1. **Periodic Strategy (Recommended for matched oracle):** Map `[*]` attachment points to periodic boundary conditions along the 1D chain axis.
2. **Oligomer Strategy (Fallback):** Construct $N$-mer oligomers (e.g., $N=3, 5$) and cap attachment points with Hydrogen (`[H]`) or Methyl (`[CH3]`) groups. Requires extrapolating HOMO-LUMO gap to $N \to \infty$.
