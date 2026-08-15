import pandas as pd
import torch
from torch.utils.data import Dataset
from encoding import one_hot_encode

class SequenceDataset(Dataset):
    def __init__(self, csv_file, split=None):
        """
        Args:
            csv_file (str): Path to the dataset csv file.
            split (str): 'train', 'val', 'test', or None for all data.
        """
        self.df = pd.read_csv(csv_file)
        if split:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.df.iloc[idx]
        seq = row['sequence']
        label = row['label']
        
        # Encode sequence
        x = one_hot_encode(seq)
        
        # Convert label to long tensor for CrossEntropyLoss
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y
