from torch import nn
from . import resnet50, VisionTransformer
import torch
from torch.functional import F


class HierarchicalEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, hidden_dim=768, pretrained=False):
        super().__init__()
        self.resnet = resnet50(pretrained)
        self.vit = VisionTransformer(img_size, patch_size, num_classes=num_classes, embed_dim=hidden_dim)
        # projection
        self.projector = nn.Linear(hidden_dim + 2048, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        # fc
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.vit(x)
        f = torch.cat([f1, f2], dim=1)
        f = self.projector(f)
        f = F.relu(self.bn1(f))
        return f, self.fc(f)

