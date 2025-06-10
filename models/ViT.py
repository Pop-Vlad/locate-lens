import math
from collections import OrderedDict

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout=0.0):
        super().__init__()
        self.mlpBlock = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        for m in self.mlpBlock.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlpBlock(x)


class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float):
        super().__init__()

        # Attention block
        self.layerNorm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.layerNorm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        y = self.layerNorm1(x)
        y = self.attention(y, y, y, need_weights=False)[0]
        y = self.dropout(y)
        # Residual connection
        x = x + y

        # MLP block
        y = self.layerNorm2(x)
        y = self.mlp(y)
        # Residual connection
        x = x + y

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, seq_length: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float,
                 attention_dropout: float):
        super().__init__()

        # Position embedding
        self.positionEmbedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)

        # Multiple layers of transformer blocks
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = TransformerBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)
        self.layers = nn.Sequential(layers)
        self.layerNorm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positionEmbedding
        return self.layerNorm(self.layers(self.dropout(x)))


class ViT(nn.Module):

    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int,
                 num_classes: int, dropout: float = 0.0, attention_dropout: float = 0.0, ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = TransformerEncoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout,
                                          attention_dropout)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)

        self.heads = nn.Sequential(heads_layers)

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.reshape(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.heads(x)
        return x
