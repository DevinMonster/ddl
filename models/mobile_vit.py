import warnings

import torch
from torch import nn
from torch.functional import F

from models.modules import InvertResidualBlock, TransformerEncoder

warnings.filterwarnings("ignore")


class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim, patch_h=2, patch_w=2, num_layer=1, num_heads=1):
        super().__init__()
        assert patch_h > 0 and patch_w > 0
        self.dim = dim
        self.L = num_layer
        self.nheads = num_heads
        self.w = patch_w
        self.h = patch_h

        # Local representations
        out1 = in_channels
        self.conv1 = nn.Conv2d(in_channels, out1, 3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out1)
        out2 = dim
        self.conv2 = nn.Conv2d(out1, out2, 1, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(out2)
        # Global representation:
        self.encoder = TransformerEncoder(num_heads, dim, dim * 2, num_layer)
        # Fusion
        out3 = in_channels
        self.conv3 = nn.Conv2d(out2, out3, 1, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(out3)
        self.conv4 = nn.Conv2d(out3 * 2, out_channels, 3, padding="same", bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def unfold(self, x):
        B, C, H, W = x.shape
        h, w = self.h, self.w
        assert H % h == 0 and W % w == 0, f"Input size must be divisible by patch size, got input size {H}*{W}, " \
                                          f"but patch size {h}*{w}"
        P = h * w
        N = H * W // P
        n_h, n_w = H // h, W // w
        # B, C, n_h, h, n_w, w
        x = x.reshape(B, C, n_h, h, n_w, w)
        # B, C, n_h, n_w, h, w
        x = x.transpose(3, 4)
        # B, C, N, P
        x = x.reshape(B, C, N, P)
        # B, P, N, C
        x = x.transpose(1, 3)
        # BP, N, C
        x = x.reshape(B * P, N, C)
        return x

    def fold(self, x, origin_shape):
        assert x.dim() == 3, f"Inputs dim must be 3, got {x.dim()}"
        B, C, H, W = origin_shape
        h, w = self.h, self.w
        P = h * w
        N = H * W // P
        n_h, n_w = H // h, W // w
        # B,P,N,C
        x = x.contiguous().view(B, P, N, C)
        # B,C,N,P
        x = x.transpose(1, 3)
        # B,C,n_h,n_w,h,w
        x = x.reshape(B, C, n_h, n_w, h, w)
        # B,C,n_h,h,n_w,w
        x = x.transpose(3, 4)
        # B,C,H,W
        x = x.reshape(origin_shape)
        return x

    def forward(self, inputs):
        # local
        x = F.hardswish(self.bn1(self.conv1(inputs)))
        x = F.hardswish(self.bn2(self.conv2(x)))

        # unfold -> BP, N, C
        shape = x.shape
        x = self.unfold(x)
        # L times transform
        x = self.encoder(x)
        # reshape back
        x = self.fold(x, shape)

        # stage4
        x = F.hardswish(self.bn3(self.conv3(x)))
        # concat at channel
        x = torch.cat((x, inputs), dim=1)
        # stage5
        x = F.hardswish(self.bn4(self.conv4(x)))
        return x


class MobileViTEncoder(nn.Module):
    num_channels = [3, 16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640]
    dims = [144, 192, 240]
    num_layers = [2, 4, 3]

    def __init__(self, in_channels=3, expansion_ratio=4):
        super().__init__()
        self.num_channels[0] = in_channels
        # conv3x3
        self.conv1 = nn.Conv2d(self.num_channels[0], self.num_channels[1], 3, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels[1])
        # 5mv2
        self.mv1 = InvertResidualBlock(self.num_channels[1], self.num_channels[2], expansion_ratio=expansion_ratio)
        self.mv2 = InvertResidualBlock(self.num_channels[2], self.num_channels[3], stride=2,
                                       expansion_ratio=expansion_ratio)
        self.mv3 = InvertResidualBlock(self.num_channels[3], self.num_channels[4], expansion_ratio=expansion_ratio)
        self.mv4 = InvertResidualBlock(self.num_channels[4], self.num_channels[5], expansion_ratio=expansion_ratio)
        self.mv5 = InvertResidualBlock(self.num_channels[5], self.num_channels[6], stride=2,
                                       expansion_ratio=expansion_ratio)
        # 2layer MobileVIT
        self.mvit1 = MobileViTBlock(self.num_channels[6], self.num_channels[7], self.dims[0],
                                    num_layer=self.num_layers[0])
        # 1mv2
        self.mv6 = InvertResidualBlock(self.num_channels[7], self.num_channels[8], stride=2,
                                       expansion_ratio=expansion_ratio)
        # 4layer MobileVIT
        self.mvit2 = MobileViTBlock(self.num_channels[8], self.num_channels[9], self.dims[1],
                                    num_layer=self.num_layers[1])
        # 1mv2
        self.mv7 = InvertResidualBlock(self.num_channels[9], self.num_channels[10], stride=2,
                                       expansion_ratio=expansion_ratio)
        # 3layer MobileVIT
        self.mvit3 = MobileViTBlock(self.num_channels[10], self.num_channels[11], self.dims[2],
                                    num_layer=self.num_layers[2])
        # point wise conv
        self.conv2 = nn.Conv2d(self.num_channels[11], self.num_channels[12], 1)
        self.bn2 = nn.BatchNorm2d(self.num_channels[12])

    def forward(self, inputs):
        x = F.hardswish(self.bn1(self.conv1(inputs)))
        x = self.mv1(x)
        x = self.mv2(x)
        x = self.mv3(x)
        x = self.mv4(x)
        x = self.mv5(x)
        x = self.mvit1(x)
        x = self.mv6(x)
        x = self.mvit2(x)
        x = self.mv7(x)
        x = self.mvit3(x)
        x = F.hardswish(self.bn2(self.conv2(x)))
        return x
