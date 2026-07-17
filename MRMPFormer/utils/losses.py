"""
NT-Xent (InfoNCE) 对比学习损失函数。
"""
import torch
import torch.nn as nn


class NT_XentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.

    每个 batch 产生 2N 个 view (N 张原图 × 2 次增强)。
    对每个 view，同一原图的另一个 view 是正样本，其余 2N-2 个是负样本。

    Args:
        temperature: 温度系数 τ (default: 0.5)，越小对难负样本越敏感
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (N, D) View A 的投影向量, L2 归一化
            z_j: (N, D) View B 的投影向量, L2 归一化

        Returns:
            scalar loss (已对 2N 取平均)
        """
        N = z_i.shape[0]
        device = z_i.device

        # 拼接所有 view: (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # 余弦相似度矩阵 (除以 τ): (2N, 2N)
        sim = torch.mm(z, z.t()) / self.temperature

        # 正样本标签: View A 的第 i 个 → View B 的第 i 个 (索引 i+N)
        labels = torch.arange(N, device=device)
        labels = torch.cat([labels + N, labels], dim=0)

        # 屏蔽自身相似度 (对角线)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float('-inf'))

        loss = self.criterion(sim, labels)
        return loss / (2 * N)
