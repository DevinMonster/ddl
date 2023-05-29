import warnings

from torch import nn
from torch.functional import F

warnings.filterwarnings("ignore")


class InvertResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_ratio=1):
        '''
        这是MobileNetv2的一个组成部分，名为invert residual block
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积核移动的步长
        :param expansion_ratio: 输出扩张率
        '''
        super(InvertResidualBlock, self).__init__()
        self.stride = stride
        # point-wise convolution
        out1 = expansion_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels, out1, kernel_size=1, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(out1)
        # depth-wise convolution
        out2 = out1
        self.conv2 = nn.Conv2d(out1, out2, 3, stride, padding=1, groups=out1, bias=False)
        self.bn2 = nn.BatchNorm2d(out2)
        # point-wise convolution
        self.conv3 = nn.Conv2d(out2, out_channels, kernel_size=1, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        x = F.hardswish(self.bn1(self.conv1(inputs)))  # first_block
        x = F.hardswish(self.bn2(self.conv2(x)))  # depth wise
        x = self.bn3(self.conv3(x))
        # residual connection
        if self.stride == 1 and x.shape == inputs.shape:
            return inputs + x
        return x


class TransformerEncoder(nn.Module):
    class MLP(nn.Module):
        def __init__(self, dim, mlp_dim, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, nheads, hidden_dim, mlp_dim, num_layer=1, dropout=0.):
        super().__init__()
        assert num_layer > 0
        assert nheads > 0 and hidden_dim % nheads == 0
        # encoder construct
        self.ln = nn.LayerNorm(hidden_dim)
        # attn = MultiHeadAttention(nheads, hidden_dim, dropout)
        attn = nn.MultiheadAttention(hidden_dim, nheads, dropout, batch_first=True)
        mlp = self.MLP(hidden_dim, mlp_dim, dropout)
        drop = nn.Dropout(dropout)
        encoder = nn.ModuleList([self.ln, attn, drop, self.ln, mlp])
        # number of layers encoder
        self.encoders = nn.ModuleList([encoder for _ in range(num_layer)])

    def forward(self, x):
        for (ln1, attn, drop, ln2, mlp) in self.encoders:
            cur = x
            x = ln1(x)
            x, _ = attn(x, x, x, need_weights=False)
            x = drop(x)
            x = x + cur

            cur = x
            x = mlp(ln2(x))
            x = x + cur
        return self.ln(x)
