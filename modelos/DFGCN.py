import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import torch.nn.functional as F
from modelos.RevIN import RevIN
from layers.GNN_variate import MultiLayerGCN_variate
from layers.GNN_time import MultiLayerGCN_time


class Model(nn.Module):
    """
    DFGCN: Dual Frequency Graph Convolutional Network
    Modelo para predicción de series de tiempo multivariadas.

    Combina dos componentes principales:
    1. GNN_time: Modela dependencias temporales mediante GCN aplicado a parches de tiempo
    2. GNN_variate: Modela dependencias entre variables usando GCN basado en correlación de Pearson
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.batch = configs.batch_size
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.multivariate = configs.enc_in
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention

        self.k = configs.k
        # Procesamiento por parches (patches)
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride + 1)
        self.embedding_patch = nn.Linear(self.patch_len, configs.d_model // 2)
        self.seq_len = self.patch_num
        self.seq_pred = nn.Linear(self.seq_len * configs.d_model // 2, configs.d_model, bias=True)
        self.PositionalEmbedding = PositionalEmbedding(configs.d_model // 2)

        self.GNN_encoder_time = MultiLayerGCN_time(
            configs.e_layers, self.d_model // 2, configs.dropout,
            configs.n_heads, configs.d_ff, self.k, configs.activation
        ).cuda()

        self.flatten = nn.Flatten(start_dim=-2)

        # Embedding de variables
        self.value_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # Normalización reversible (RevIN)
        revin = True
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(self.multivariate, affine=True, subtract_last=False)

        self.GNN_encoder = MultiLayerGCN_variate(
            configs.e_layers, self.d_model, configs.dropout,
            configs.n_heads, configs.d_ff, self.k, configs.activation
        ).cuda()

        self.FC = nn.Linear(configs.d_model * 1, configs.pred_len)
        self.FC2 = nn.Linear(configs.d_model * 1, configs.pred_len)
        self.FC3 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def Embedding_patch(self, seasonal_init, N):
        x_enc_patch_row = seasonal_init.transpose(2, 1).unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        x_enc_patch = self.embedding_patch(x_enc_patch_row)
        x_enc_patch = x_enc_patch.reshape(-1, self.patch_num, self.d_model // 2)
        x_enc_patch = self.PositionalEmbedding(x_enc_patch) + x_enc_patch
        x_enc_patch = x_enc_patch.reshape(-1, N, self.patch_num, self.d_model // 2)
        return x_enc_patch, x_enc_patch_row

    def Channel_independence(self, enc_in, x_enc, B, x_enc_patch_row):
        enc_out_in = torch.mean(enc_in, dim=1, keepdim=False)
        dec_out_time = self.GNN_encoder_time(enc_out_in, enc_out_in, enc_in)
        enc_out_in = self.flatten(dec_out_time)
        dec_out_time = self.seq_pred(enc_out_in)
        return dec_out_time

    def forecast(self, x_enc):
        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')

        B, _, N = x_enc.shape

        # Rama temporal (Time GNN)
        enc_out_in, x_enc_patch_row = self.Embedding_patch(x_enc, N)
        dec_out_time = self.Channel_independence(enc_out_in, x_enc, B, x_enc_patch_row)

        # Rama de variables (Variate GNN)
        enc_out_vari = self.value_embedding(x_enc.transpose(2, 1))
        dec_out_vari = self.GNN_encoder(enc_out_vari, enc_out_vari.transpose(2, 1))

        # Fusión de ambas ramas
        dec_out_time = self.FC(dec_out_time)
        dec_out_vari = self.FC2(dec_out_vari)
        enc_out_concat = torch.cat((dec_out_time, dec_out_vari), dim=-1)
        dec_out = self.FC3(enc_out_concat)

        if self.use_norm:
            dec_out_all = self.revin_layer(dec_out.transpose(2, 1), 'denorm')
        else:
            dec_out_all = dec_out.transpose(2, 1)

        return dec_out_all

    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
