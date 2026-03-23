import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn as nn
import numpy as np
from layers.Transformer_encoder import TransformerEncoder


class MultiLayerGCN_variate(nn.Module):
    """
    GCN multi-capa para modelar dependencias entre variables (variates).
    Construye un grafo dinámico basado en correlación de Pearson entre series.
    """

    def __init__(self, num_layers, d_model, dropout, n_heads, d_ff, k, activation):
        super(MultiLayerGCN_variate, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1):
            self.layers.append(
                GCN(d_model, d_model, d_model, dropout, n_heads, d_ff, num_layers, activation)
            )
        self.d_model = d_model
        self.k = k

    def pearson_correlation(self, x):
        batch_size, num_vars, input_len = x.shape
        mean = x.mean(dim=2, keepdim=True)
        centered_data = x - mean
        cov_matrix = torch.bmm(centered_data, centered_data.transpose(1, 2)) / (input_len - 1)
        std_dev = torch.sqrt(torch.diagonal(cov_matrix, dim1=-2, dim2=-1))
        std_dev[std_dev == 0] = 1
        std_dev_matrix = std_dev.unsqueeze(2) * std_dev.unsqueeze(1)
        correlation_matrix = cov_matrix / std_dev_matrix
        return correlation_matrix

    def edge_index(self, x):
        similarity_matrix = self.pearson_correlation(x)
        batch_size, num_nodes = similarity_matrix.shape[:2]

        # Fix: clamp k so it cannot exceed num_nodes - 1; with a single variable
        # there are no valid neighbors and we fall back to a self-loop.
        k_actual = min(self.k, num_nodes - 1)

        if num_nodes == 1:
            # Fix: cannot build a k-NN graph with only 1 node; use a self-loop
            # so GCNConv can still propagate features without crashing.
            self_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)
            edge_index = torch.stack([self_idx, self_idx], dim=1)  # (B, 2, 1)
            return edge_index

        neighbors = torch.argsort(similarity_matrix, dim=-1)[:, :, 1:k_actual + 1]
        row_indices = torch.arange(num_nodes, device=x.device).repeat(k_actual).reshape(1, -1).repeat(batch_size, 1)
        col_indices = neighbors.reshape(batch_size, -1)
        edge_index = torch.stack((row_indices, col_indices), dim=1)
        # Fix: use .to(x.device) instead of .cuda() to support CPU-only environments
        edge_index = edge_index.long().to(x.device)
        return edge_index

    def forward(self, enc_out_vari, x_enc):
        edge_index = self.edge_index(x_enc.transpose(2, 1))
        data_list = [Data(x=enc_out_vari[i], edge_index=edge_index[i])
                     for i in range(enc_out_vari.size(0))]
        batch = Batch.from_data_list(data_list)

        x_raw = batch.x
        edge_index = batch.edge_index

        for layer in self.layers:
            x = layer(enc_out_vari, x_enc, x_raw, edge_index)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, n_heads, d_ff, num_layers, activation):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels).cuda()
        self.conv2 = GCNConv(hidden_channels, hidden_channels).cuda()
        self.dropout = nn.Dropout(dropout)
        self.Transformer_encoder_2 = TransformerEncoder(
            in_channels, n_heads, num_layers, d_ff, dropout
        ).cuda()

        if activation == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

        self.norm1 = nn.LayerNorm(in_channels)

    def forward(self, enc_out_vari, x_enc, x_raw, edge_index):
        B, M, d_model = enc_out_vari.size()

        x1 = self.conv1(x_raw, edge_index)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)

        x = x2.reshape(B, M, d_model)

        # x como V, K; enc_out_vari como Q
        enc_out_vari_trans = self.Transformer_encoder_2(enc_out_vari, x, mask=None)

        return enc_out_vari_trans
