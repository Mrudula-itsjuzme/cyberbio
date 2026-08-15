import torch
import pandas as pd
from model import MotifCNN
from encoding import one_hot_encode
from data_generator import generate_background_sequence
import numpy as np

device = "cpu"
model = MotifCNN().to(device)
model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
model.eval()

df = pd.read_csv("../data/raw/dataset.csv")
df = df[df['split'] == 'test']

motif_0 = "ACGTACGT"
motif_1 = "TGCATGCA"
alphabet = ['A', 'C', 'G', 'T']

def predict_prob(seq):
    t = one_hot_encode(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        prob = torch.softmax(out, dim=1)[0, 1].item()
    return prob

c0_A_probs = []
c0_B_probs = []
c0_D_probs = []

c1_A_probs = []
c1_B_probs = []
c1_E_probs = []

for idx, row in df[df['label'] == 0].head(100).iterrows():
    seq = row['sequence']
    idx_0 = seq.find(motif_0)
    c0_A_probs.append(predict_prob(seq))
    
    bg = generate_background_sequence(8, alphabet, [motif_0, motif_1])
    seq_b = seq[:idx_0] + bg + seq[idx_0+8:]
    c0_B_probs.append(predict_prob(seq_b))
    
    inject_pos = (idx_0 + 13) % 42
    seq_d = seq[:inject_pos] + motif_1 + seq[inject_pos+8:]
    c0_D_probs.append(predict_prob(seq_d))

for idx, row in df[df['label'] == 1].head(100).iterrows():
    seq = row['sequence']
    idx_1 = seq.find(motif_1)
    c1_A_probs.append(predict_prob(seq))
    
    bg = generate_background_sequence(8, alphabet, [motif_0, motif_1])
    seq_b = seq[:idx_1] + bg + seq[idx_1+8:]
    c1_B_probs.append(predict_prob(seq_b))
    
    inject_pos = (idx_1 + 13) % 42
    seq_e = seq[:inject_pos] + motif_0 + seq[inject_pos+8:]
    c1_E_probs.append(predict_prob(seq_e))

print(f"Class 0 - A: {np.mean(c0_A_probs):.4f}, B (Motif removed): {np.mean(c0_B_probs):.4f}, D (Opp added): {np.mean(c0_D_probs):.4f}")
print(f"Class 1 - A: {np.mean(c1_A_probs):.4f}, B (Motif removed): {np.mean(c1_B_probs):.4f}, E (Opp added): {np.mean(c1_E_probs):.4f}")
