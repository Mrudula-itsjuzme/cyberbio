import torch
import torch.nn as nn
import torch.nn.functional as F

class MotifCNN(nn.Module):
    """
    A simple 1D Convolutional Neural Network for binary classification of DNA sequences.
    
    The architecture is designed to be simple and easy to understand:
    1. A 1D convolutional layer to detect local sequence patterns (motifs).
    2. A max pooling layer to aggregate pattern detections across the sequence.
    3. Fully connected layers for classification.
    """
    def __init__(self, sequence_length=50, num_classes=2):
        super(MotifCNN, self).__init__()
        
        # Input shape: (batch_size, 4 channels, sequence_length)
        # kernel_size=8 is chosen because our synthetic motifs are 8bp long
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8, bias=False)
        self.bn1 = nn.BatchNorm1d(16, affine=False)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        
        # Calculate the size after conv and pool
        # L_out_conv = 50 - 8 + 1 = 43
        # L_out_pool = 43 // 2 = 21
        self.fc_input_size = 16 * 21
        
        self.fc1 = nn.Linear(self.fc_input_size, 32, bias=False)
        self.fc2 = nn.Linear(32, num_classes, bias=False)
        
    def forward(self, x):
        # Apply convolution, batch norm, and ReLU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Apply max pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        
        # Output layer (raw logits for BCEWithLogitsLoss)
        x = self.fc2(x)
        
        return x
