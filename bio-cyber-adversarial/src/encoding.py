import torch

def one_hot_encode(sequence):
    """
    One-hot encodes a DNA sequence.
    A = [1, 0, 0, 0]
    C = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    T = [0, 0, 0, 1]
    
    Returns a PyTorch tensor of shape (4, len(sequence)) suitable for 1D CNNs.
    """
    mapping = {
        'A': [1.0, 0.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'T': [0.0, 0.0, 0.0, 1.0]
    }
    
    # If a character is not in the mapping, use all zeros
    encoded = [mapping.get(char.upper(), [0.0, 0.0, 0.0, 0.0]) for char in sequence]
    
    # Convert to tensor. Shape becomes (sequence_length, 4)
    tensor = torch.tensor(encoded, dtype=torch.float32)
    
    # Transpose to (4, sequence_length) for PyTorch 1D convolutions (channels, length)
    tensor = tensor.transpose(0, 1)
    
    return tensor
