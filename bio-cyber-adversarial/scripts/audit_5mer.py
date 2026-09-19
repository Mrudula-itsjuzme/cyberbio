import pandas as pd
df = pd.read_csv("data/v2/dataset.csv")

def count_composite(seq):
    # The composite 5-mers spanning ATGC and GCAT when dist=5
    # pos1: A, T, G, C
    # pos1+1: T, G, C, N
    # pos1+2: G, C, N, G
    # pos1+3: C, N, G, C
    # pos1+4: N, G, C, A
    composites = 0
    for N in "ACGT":
        composites += seq.count("TGC" + N + "G")
        composites += seq.count("GC" + N + "GC")
        composites += seq.count("C" + N + "GCA")
    return composites

df["composites"] = df["sequence"].apply(count_composite)
c0 = df[df["label"] == 0]["composites"].sum()
c1 = df[df["label"] == 1]["composites"].sum()

print(f"Composite 5-mers in Class 0: {c0}")
print(f"Composite 5-mers in Class 1: {c1}")
