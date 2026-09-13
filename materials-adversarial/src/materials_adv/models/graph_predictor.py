import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_node = nn.Linear(in_dim, out_dim)
        self.linear_msg = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj, mask):
        # x: [B, N, F], adj: [B, N, N]
        # Message passing: sum of neighbor features
        # Normalize adjacency by degree
        degree = adj.sum(dim=-1, keepdim=True)
        adj_norm = adj / (degree + 1e-8)
        
        msg = torch.bmm(adj_norm, x) # [B, N, F]
        out = self.linear_node(x) + self.linear_msg(msg)
        out = F.relu(out)
        out = out * mask.unsqueeze(-1)
        return out

class GraphPredictor(nn.Module):
    def __init__(self, node_dim=7, hidden_dim=64, num_layers=3):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Compact MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def encode(self, x, adj, mask):
        h = self.node_embed(x)
        for layer in self.layers:
            h = layer(h, adj, mask)
        
        # Global mean pooling
        # sum over nodes / num_nodes
        num_nodes = mask.sum(dim=1, keepdim=True).clamp(min=1)
        h_pool = h.sum(dim=1) / num_nodes
        return h_pool
        
    def forward(self, x, adj, mask):
        h_pool = self.encode(x, adj, mask)
        return self.head(h_pool)
