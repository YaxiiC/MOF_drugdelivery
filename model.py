# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class GINToxModel(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn=mlp, edge_dim=num_edge_features)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [N, num_node_features]
        # edge_index: [2, E]
        # edge_attr: [E, num_edge_features]
        # batch: [N] graph indices

        x = self.input_proj(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling to get graph-level embedding
        x = global_add_pool(x, batch)  # [num_graphs, hidden_dim]

        out = self.readout_mlp(x)  # [num_graphs, num_classes]
        return out