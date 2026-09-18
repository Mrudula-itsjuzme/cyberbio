import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import MotifCNN
from encoding import one_hot_encode

def get_best_substitution(model, sequence_tensor, target_label, mask):
    model.eval()
    x = sequence_tensor.clone().unsqueeze(0)
    x.requires_grad = True
    
    output = model(x)
    loss_fn = nn.CrossEntropyLoss()
    target = torch.tensor([target_label], dtype=torch.long).to(output.device)
    loss = loss_fn(output, target)
    
    model.zero_grad()
    loss.backward()
    
    grad = x.grad.squeeze(0)
    current_bases = sequence_tensor.argmax(dim=0)
    
    best_delta = 0.0
    best_i = -1
    best_b = -1
    
    for i in range(sequence_tensor.shape[1]):
        if mask[i] == 0:
            continue
        c = current_bases[i]
        for b in range(4):
            if b == c:
                continue
            delta = grad[b, i] - grad[c, i]
            if delta < best_delta:
                best_delta = delta.item()
                best_i = i
                best_b = b
                
    return best_i, best_b

def tensor_to_string(tensor):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    idx = tensor.argmax(dim=0).tolist()
    return "".join(mapping[i] for i in idx)

def attack_sequence(model, sequence_str, true_label, motif_pos, motif_len=8, max_iters=20):
    target_label = 1 - true_label
    tensor = one_hot_encode(sequence_str)
    
    mask = [1] * len(sequence_str)
    for i in range(motif_pos, motif_pos + motif_len):
        mask[i] = 0
        
    model.eval()
    
    for _ in range(max_iters):
        with torch.no_grad():
            out = model(tensor.unsqueeze(0))
            pred_label = out.argmax(dim=1).item()
            
        if pred_label == target_label:
            return tensor_to_string(tensor), True, _
            
        best_i, best_b = get_best_substitution(model, tensor, target_label, mask)
        if best_i == -1:
            break
            
        tensor[:, best_i] = 0
        tensor[best_b, best_i] = 1.0
        
    return tensor_to_string(tensor), False, max_iters

def generate_adversarial_dataset(model_path, data_path, output_path, max_iters=20):
    df = pd.read_csv(data_path)
    test_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
    
    model = MotifCNN()
    model.load_state_dict(torch.load(model_path))
    
    successes = 0
    adv_sequences = []
    
    print(f"Attacking {len(test_df)} test sequences...")
    for idx, row in test_df.iterrows():
        seq = row['sequence']
        label = row['label']
        motif_pos = row['motif_pos']
        
        adv_seq, success, iters = attack_sequence(model, seq, label, motif_pos, motif_len=8, max_iters=max_iters)
        
        adv_sequences.append(adv_seq)
        if success:
            successes += 1
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} - Current Success Rate: {successes / (idx + 1) * 100:.2f}%")
            
    test_df['adv_sequence'] = adv_sequences
    
    print(f"Final Attack Success Rate: {successes / len(test_df) * 100:.2f}%")
    test_df.to_csv(output_path, index=False)
    print(f"Saved adversarial test set to {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "models", "best_model.pth")
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    output_path = os.path.join(script_dir, "..", "data", "raw", "adv_dataset.csv")
    generate_adversarial_dataset(model_path, data_path, output_path, max_iters=10)
