import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import torch.nn as nn
import numpy as np
from layers.Transformer_encoder import TransformerEncoder
import torch.nn.functional as F


class MultiLayerGCN_time(nn.Module):
    """
    GCN multi-capa para modelar dependencias temporales.
    Construye un grafo dinámico basado en correlación de Pearson entre parches de tiempo.
    """

    def __init__(self, num_layers, d_model, dropout, n_heads, d_ff, k, activation):
        super(MultiLayerGCN_time, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1):
            self.layers.append(
                GCN(d_model, d_model, d_model, dropout, n_heads, d_ff, 1, activation)
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
        k = 2
        neighbors = torch.argsort(similarity_matrix, dim=-1)[:, :, 1:k + 1]
        batch_size, num_nodes = similarity_matrix.shape[:2]
        row_indices = torch.arange(num_nodes, device=x.device).repeat(k).reshape(1, -1).repeat(batch_size, 1)
        col_indices = neighbors.reshape(batch_size, -1)
        edge_index = torch.stack((row_indices, col_indices), dim=1)
        edge_index = edge_index.long().cuda()
        return edge_index

    def forward(self, enc_out_vari_embeding, x_enc, enc_in):
        edge_index = self.edge_index(x_enc)
        data_list = [Data(x=enc_out_vari_embeding[i], edge_index=edge_index[i])
                     for i in range(enc_out_vari_embeding.size(0))]
        batch = Batch.from_data_list(data_list)

        x_raw = batch.x
        edge_index = batch.edge_index

        for layer in self.layers:
            x = layer(enc_out_vari_embeding, enc_in, x_raw, edge_index)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, n_heads, d_ff, num_layers, activation):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.Transformer_encoder_time = TransformerEncoder(
            in_channels, n_heads, num_layers, d_ff, dropout
        ).cuda()

        if activation == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

        self.norm1 = nn.LayerNorm(in_channels)

    def forward(self, enc_out_time, enc_in, x_raw, edge_index):
        B, M, patch_num, d_model = enc_in.size()

        x1 = self.conv1(x_raw, edge_index)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)
        x = x2.reshape(-1, patch_num, d_model)

        x = x.reshape(B, patch_num, d_model).unsqueeze(1).expand(-1, M, -1, -1) + enc_in
        x = x.reshape(-1, patch_num, d_model)
        enc_out_vari_trans = self.Transformer_encoder_time(x, x, mask=None)
        enc_out_vari_trans = enc_out_vari_trans.reshape(B, M, patch_num, d_model)

        return enc_out_vari_trans
