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


def alignment_loss(z_i, z_j):
    """
    Alignment: 正样本对之间的平均平方 L2 距离。

    测量同一图像的两个增强视图在特征空间中多接近。值越小越好。
    参考: Wang & Isola, "Understanding Contrastive Learning", ICML 2020.

    Args:
        z_i: (N, D) View A, L2 归一化
        z_j: (N, D) View B, L2 归一化

    Returns:
        scalar, 范围 [0, 4]（L2 归一化向量差的平方最大为 4）
    """
    return (z_i - z_j).pow(2).sum(dim=1).mean()


def uniformity_loss(z, t=2.0):
    """
    Uniformity: 特征在单位球面上分布的均匀程度。

    测量所有特征向量之间高斯势的平均对数。值越小（越负）越均匀。
    对于 128-d 单位球面，理论最优 ≈ -4.0。
    参考: Wang & Isola, "Understanding Contrastive Learning", ICML 2020.

    Args:
        z: (N, D) L2 归一化特征向量
        t: 高斯核带宽 (default: 2.0)

    Returns:
        scalar, 越负越好
    """
    N = z.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=z.device)
    dist2 = torch.cdist(z, z, p=2).pow(2)           # (N, N) 平方欧氏距离
    mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    potentials = torch.exp(-t * dist2[mask])          # 排除对角线
    return torch.log(potentials.mean())
