import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import pandas as pd
import random

from dataset import SequenceDataset
from model import MotifCNN
from encoding import one_hot_encode
from data_generator import generate_background_sequence

def evaluate_model(data_path="../data/raw/dataset.csv", model_path="../models/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MotifCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_dataset = SequenceDataset(data_path, split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Evaluating test set...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(batch_y.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("\n--- Strengthened Interpretability Sanity Check ---")
    
    motif_0 = "ACGTACGT"
    motif_1 = "TGCATGCA"
    alphabet = ['A', 'C', 'G', 'T']
    
    df = pd.read_csv(data_path)
    
    def predict_prob(seq):
        t = one_hot_encode(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            prob = torch.softmax(out, dim=1)[0, 1].item()
        return prob
    
    # Evaluate 100 samples for stable interpretability metrics
    c0_A_probs, c0_B_probs, c0_C_probs, c0_D_probs = [], [], [], []
    for idx, row in df[(df['label'] == 0) & (df['split'] == 'test')].head(100).iterrows():
        seq = row['sequence']
        idx_0 = seq.find(motif_0)
        c0_A_probs.append(predict_prob(seq))
        
        bg = generate_background_sequence(len(motif_0), alphabet, [motif_0, motif_1])
        c0_B_probs.append(predict_prob(seq[:idx_0] + bg + seq[idx_0+len(motif_0):]))
        
        bg_full = generate_background_sequence(len(seq), alphabet, [motif_0, motif_1])
        c0_C_probs.append(predict_prob(bg_full[:idx_0] + motif_0 + bg_full[idx_0+len(motif_0):]))
        
        inject_pos = (idx_0 + len(motif_0) + 5) % (len(seq) - len(motif_1))
        if abs(inject_pos - idx_0) < len(motif_0): inject_pos = len(seq) - len(motif_1)
        c0_D_probs.append(predict_prob(seq[:inject_pos] + motif_1 + seq[inject_pos+len(motif_1):]))

    c1_A_probs, c1_B_probs, c1_C_probs, c1_E_probs = [], [], [], []
    for idx, row in df[(df['label'] == 1) & (df['split'] == 'test')].head(100).iterrows():
        seq = row['sequence']
        idx_1 = seq.find(motif_1)
        c1_A_probs.append(predict_prob(seq))
        
        bg = generate_background_sequence(len(motif_1), alphabet, [motif_0, motif_1])
        c1_B_probs.append(predict_prob(seq[:idx_1] + bg + seq[idx_1+len(motif_1):]))
        
        bg_full = generate_background_sequence(len(seq), alphabet, [motif_0, motif_1])
        c1_C_probs.append(predict_prob(bg_full[:idx_1] + motif_1 + bg_full[idx_1+len(motif_1):]))
        
        inject_pos = (idx_1 + len(motif_1) + 5) % (len(seq) - len(motif_0))
        if abs(inject_pos - idx_1) < len(motif_1): inject_pos = len(seq) - len(motif_0)
        c1_E_probs.append(predict_prob(seq[:inject_pos] + motif_0 + seq[inject_pos+len(motif_0):]))
        
    import numpy as np
    p0_A = np.mean(c0_A_probs)
    p0_B = np.mean(c0_B_probs)
    p0_C = np.mean(c0_C_probs)
    p0_D = np.mean(c0_D_probs)
    
    p1_A = np.mean(c1_A_probs)
    p1_B = np.mean(c1_B_probs)
    p1_C = np.mean(c1_C_probs)
    p1_E = np.mean(c1_E_probs)
    
    print("\nModel Behavior Report (Probability of Class 1) - Average over 100 test samples:")
    print(f"{'':<20} | {'Original (A)':<15} | {'Motif Altered (B)':<20} | {'BG Altered (C)':<15} | {'Opposite Added (D/E)':<20}")
    print("-" * 100)
    
    print(f"{'Class 0 sample':<20} | {p0_A:<15.4f} | {p0_B:<20.4f} | {p0_C:<15.4f} | {p0_D:<20.4f}")
    print(f"{'Class 1 sample':<20} | {p1_A:<15.4f} | {p1_B:<20.4f} | {p1_C:<15.4f} | {p1_E:<20.4f}")
    
    print("\nAnalysis:")
    if p1_B < 0.2 or p0_B > 0.8:
        print("-> Motif Altered (B) is pushing probabilities toward the opposite class, indicating the shortcut might still exist.")
    elif 0.3 <= p1_B <= 0.7 and 0.3 <= p0_B <= 0.7:
        print("-> Motif Altered (B) results in ~0.5 probability (confusion). The model correctly relies on BOTH motifs!")
        
    if 0.3 <= p1_E <= 0.7 and 0.3 <= p0_D <= 0.7:
        print("-> Opposite Added (D/E) results in ~0.5 probability (confusion). The model correctly weighs both pieces of evidence!")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    model_path = os.path.join(script_dir, "..", "models", "best_model.pth")
    
    evaluate_model(data_path=data_path, model_path=model_path)
