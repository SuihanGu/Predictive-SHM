# -*- coding: utf-8 -*-
"""
Transformer-CNN 模型结构（与 prediction_service / 训练脚本一致）
供 app.adapters.model_adapter 与 models.train 使用。
"""
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_hidden_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.self_ff = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, input_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.self_ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Conv1dLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dropout):
        super(Conv1dLayer, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.dropout(self.relu(x))
        x = x.permute(0, 2, 1)
        return x


class TransformerCnn(nn.Module):
    def __init__(
        self,
        response_dim,
        env_dim,
        trans_dim,
        num_heads,
        ff_hidden_dim,
        conv_hidden_dim,
        kernel_size,
        dropout,
        n_steps,
        lag,
        m,
    ):
        super(TransformerCnn, self).__init__()
        self.response_dim = response_dim
        self.n_steps = n_steps
        self.transformer = TransformerEncoderLayer(
            input_dim=trans_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout,
        )
        self.conv_response = Conv1dLayer(
            input_dim=response_dim,
            output_dim=conv_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.conv_env = Conv1dLayer(
            input_dim=env_dim,
            output_dim=conv_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.conv_trans = Conv1dLayer(
            input_dim=trans_dim,
            output_dim=conv_hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.final_conv = Conv1dLayer(
            input_dim=conv_hidden_dim,
            output_dim=response_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        fc_input_dim = response_dim * (lag + m * 2)
        self.fc = nn.Linear(fc_input_dim, response_dim * n_steps)

    def forward(self, x_response, x_env, x_cat):
        x_cat = self.transformer(x_cat)
        x_response_conv = self.conv_response(x_response)
        x_env_conv = self.conv_env(x_env)
        x_cat_conv = self.conv_trans(x_cat)
        x_concat = torch.cat([x_response_conv, x_env_conv, x_cat_conv], dim=1)
        x_final_conv = self.final_conv(x_concat)
        x_final_flat = x_final_conv.reshape(x_final_conv.size(0), -1)
        x_final_fc = self.fc(x_final_flat)
        output = x_final_fc.view(x_final_conv.size(0), -1, self.response_dim)
        return output
