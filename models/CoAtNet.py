import torch.nn as nn
from einops.layers.torch import Rearrange

from models.ViT import MLP


class SqueezeAndExcitationBlock(nn.Module):

    def __init__(self, input, output, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(output, int(input * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(input * expansion), output, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, num_channels)
        y = self.net(y).view(batch_size, num_channels, 1, 1)
        return x * y


class MBConvBlock(nn.Module):  # Mobile Inverted Residual Bottleneck Block

    def __init__(self, input, output, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        if not self.downsample:
            stride = 1
        else:
            stride = 2
        hidden_dim = int(input * expansion)

        if self.downsample:
            self.maxPool = nn.MaxPool2d(3, 2, 1)
            self.conv2d = nn.Conv2d(input, output, 1, 1, 0, bias=False)

        if expansion == 1:
            self.convBlock = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output),
            )
        else:
            self.convBlock = nn.Sequential(
                nn.Conv2d(input, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SqueezeAndExcitationBlock(input, hidden_dim),
                nn.Conv2d(hidden_dim, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output),
            )

        self.batchNorm = nn.BatchNorm2d(input)

    def forward(self, x):
        if self.downsample:
            return self.conv2d(self.maxPool(x)) + self.convBlock(self.batchNorm(x))
        else:
            return x + self.convBlock(x)


class TransformerBlock(nn.Module):  # Transformer Block
    def __init__(self, input_size, output_size, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()

        self.downsample = downsample
        if self.downsample:
            self.maxPool1 = nn.MaxPool2d(3, 2, 1)
            self.maxPool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(input_size, output_size, 1, 1, 0, bias=False)

        # Attention Block
        self.flatten = Rearrange('b c ih iw -> b (ih iw) c')
        self.layerNorm1 = nn.LayerNorm(input_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(input_size, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.reshape = Rearrange('b (ih iw) c -> b c ih iw', ih=image_size[0], iw=image_size[1])

        # MLP Block
        self.layerNorm2 = nn.LayerNorm(output_size)
        self.mlp = MLP(output_size, int(input_size * 4), dropout)
        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            self.layerNorm2,
            self.mlp,
            Rearrange('b (ih iw) c -> b c ih iw', ih=image_size[0], iw=image_size[1])
        )

    def forward(self, x):
        if self.downsample:
            y = self.flatten(self.maxPool2(x))
            y = self.layerNorm1(y)
            y, _ = self.self_attention(y, y, y)
            y = self.dropout(y)
            y = self.reshape(y)
            x = self.proj(self.maxPool1(x) + y)
        else:
            y = self.flatten(x)
            y = self.layerNorm1(y)
            y, _ = self.self_attention(y, y, y)
            y = self.dropout(y)
            y = self.reshape(y)
            x = x + y
        x = x + self.ff(x)
        return x


def first_convolutional_bock(input, output, image_size, downsample=False):
    if not downsample:
        stride = 1
    else:
        stride = 2
    return nn.Sequential(
        nn.Conv2d(input, output, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output),
        nn.GELU()
    )


class CoAtNet(nn.Module):  # CoAtNet Model
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000,
                 block_types=['C', 'C', 'T', 'T']):  # C: MBConv Block, T: Transformer Block
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConvBlock, 'T': TransformerBlock}

        self.s0 = self.buildLayer(
            first_convolutional_bock, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self.buildLayer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self.buildLayer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self.buildLayer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self.buildLayer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=True)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        x = (x + 100) / 200  # Scaling output improves performance
        return x

    def buildLayer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)
