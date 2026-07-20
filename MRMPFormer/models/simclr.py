"""
SimCLR model: ResNet50 backbone + MLP projection head.

支持分阶段冻结 backbone:
  freeze_stages=0 → 全部可训练（默认）
  freeze_stages=4 → 冻结 stem+layer1+layer2+layer3，仅训练 layer4 + 投影头
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
        freeze_stages: 冻结前 N 个 stage (0~5)
            0 = 全部可训练
            1 = 冻结 stem
            2 = 冻结 stem + layer1
            3 = 冻结 stem + layer1 + layer2
            4 = 冻结 stem + layer1 + layer2 + layer3（仅训练 layer4 + 投影头）
            5 = 冻结全部 backbone（线性探针）
    """

    def __init__(self, proj_hidden_dim=512, proj_output_dim=128, pretrained=True,
                 freeze_stages=0):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.backbone_dim = resnet.fc.in_features  # 2048

        # 拆分 backbone 为独立组件，支持按 stage 冻结
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 保留整体 Sequential 以兼容外部直接调用 self.backbone(x)
        self.backbone = nn.Sequential(
            self.stem, self.layer1, self.layer2,
            self.layer3, self.layer4, self.avgpool,
        )

        self.projection = ProjectionHead(self.backbone_dim, proj_hidden_dim, proj_output_dim)

        # 执行冻结
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, n):
        """冻结前 n 个 stage 的参数 (stem=0, layer1=1, layer2=2, layer3=3, layer4=4)。"""
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(n, len(stages))):
            for param in stages[i].parameters():
                param.requires_grad = False

    def trainable_param_counts(self):
        """返回 (可训练参数量, 总参数量) 用于日志输出。"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

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
