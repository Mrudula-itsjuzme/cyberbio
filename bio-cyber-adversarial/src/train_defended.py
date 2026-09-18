import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from encoding import one_hot_encode
from model import MotifCNN

class DefendedSequenceDataset(Dataset):
    def __init__(self, csv_file, split=None):
        self.df = pd.read_csv(csv_file)
        if split:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['sequence']
        label = row['label']
        motif_pos = row['motif_pos']
        
        x = one_hot_encode(seq)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y, motif_pos

def generate_adv_batch(model, x, y, motif_positions, motif_len=8, iters=1):
    x_adv = x.clone().detach()
    
    for _ in range(iters):
        model.eval()
        x_adv.requires_grad = True
        outputs = model(x_adv)
        
        loss_fn = nn.CrossEntropyLoss()
        target_y = 1 - y
        loss = loss_fn(outputs, target_y)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad
        current_bases = x_adv.argmax(dim=1)
        
        x_new = x_adv.clone().detach()
        for b in range(x.shape[0]):
            best_delta = 0.0
            best_i = -1
            best_base = -1
            c_bases = current_bases[b]
            pos = motif_positions[b].item()
            
            for i in range(x.shape[2]):
                if pos <= i < pos + motif_len:
                    continue
                c = c_bases[i]
                for base in range(4):
                    if base == c:
                        continue
                    delta = grad[b, base, i] - grad[b, c, i]
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_base = base
                        
            if best_i != -1:
                x_new[b, :, best_i] = 0
                x_new[b, best_base, best_i] = 1.0
                
        x_adv = x_new
        
    model.train()
    return x_adv

def train_defended(data_path="../data/raw/dataset.csv", model_dir="../models", epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = DefendedSequenceDataset(data_path, split="train")
    val_dataset = DefendedSequenceDataset(data_path, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MotifCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    
    print("Starting adversarial training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y, batch_pos in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            batch_x_adv = generate_adv_batch(model, batch_x, batch_y, batch_pos, motif_len=8, iters=2)
            
            combined_x = torch.cat([batch_x, batch_x_adv], dim=0)
            combined_y = torch.cat([batch_y, batch_y], dim=0)
            
            optimizer.zero_grad()
            outputs = model(combined_x)
            loss = criterion(outputs, combined_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * combined_x.size(0)
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == combined_y).sum().item()
            train_total += combined_x.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_pos in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_x.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(model_dir, "defended_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best defended model to {save_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    model_dir = os.path.join(script_dir, "..", "models")
    train_defended(data_path=data_path, model_dir=model_dir)
