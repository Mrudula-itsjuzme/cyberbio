import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dataset import SequenceDataset
from model import MotifCNN

def train(data_path="../data/raw/dataset.csv", model_dir="../models", epochs=10, batch_size=32, lr=0.001):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = SequenceDataset(data_path, split="train")
    val_dataset = SequenceDataset(data_path, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = MotifCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    os.makedirs(model_dir, exist_ok=True)
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_x.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
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
            save_path = os.path.join(model_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model to {save_path}")

if __name__ == "__main__":
    # Calculate path relative to the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    model_dir = os.path.join(script_dir, "..", "models")
    
    train(data_path=data_path, model_dir=model_dir)
