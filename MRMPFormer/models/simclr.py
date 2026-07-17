"""
SimCLR model: ResNet50 backbone + MLP projection head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    """
    SimCLR 对比学习模型。

    Args:
        proj_hidden_dim: 投影头隐藏层维度 (default: 512)
        proj_output_dim: 投影头输出维度 (default: 128)
        pretrained: 是否加载 ImageNet 预训练权重 (default: True)
    """

    def __init__(self, proj_hidden_dim=512, proj_output_dim=128, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.backbone_dim = resnet.fc.in_features  # 2048
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.projection = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

    def forward(self, x):
        """训练前向：输出 L2 归一化的投影特征 (B, output_dim)。"""
        h = self.backbone(x)
        h = h.flatten(1)
        z = self.projection(h)
        return F.normalize(z, dim=1)

    def get_features(self, x):
        """推理用：输出 backbone 特征 (B, 2048)，无投影头。"""
        h = self.backbone(x)
        return h.flatten(1)
