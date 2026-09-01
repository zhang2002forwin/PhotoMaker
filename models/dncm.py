"""DNCM (Deterministic Neural Color Mapping，确定性神经颜色映射)。

实现 Neural Preset 的核心颜色映射:
    Y = I(h*w, 3) @ P(3, k) @ T(k, k) @ Q(k, 3)

其中:
  - P, Q 是所有图像共享的可学习投影矩阵 (与图像无关)
  - T 是由编码器 E 从缩略图预测的图像自适应矩阵，仅 k*k 个参数
  - 映射是确定性的: 相同输入颜色 -> 相同输出颜色，避免伪影
  - 内存高效: 每张图像仅需 k*k 个自适应参数 (k=16 时仅 256 个)
  - 逐像素独立运算，支持 4K/8K 高分辨率图像
"""
import torch
import torch.nn as nn


class DNCM(nn.Module):
    """确定性神经颜色映射模块。

    Args:
        k: 瓶颈维度，T 矩阵为 k*k。
    """

    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k

        # P 和 Q 是所有图像共享的投影矩阵 (与图像无关)
        # 用小随机值初始化，使初始映射接近恒等
        self.P = nn.Parameter(torch.randn(3, k) * 0.01)
        self.Q = nn.Parameter(torch.randn(k, 3) * 0.01)

    def forward(self, image: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """应用确定性颜色映射。

        Args:
            image: (B, 3, H, W) 输入图像，范围 [0, 1]
            T:     (B, k, k) 图像自适应变换矩阵

        Returns:
            output: (B, 3, H, W) 映射后的图像
        """
        B, C, H, W = image.shape
        assert C == 3, "DNCM 需要 3 通道 RGB 图像"

        # 展平空间维度: (B, 3, H*W) -> (B, H*W, 3)
        x = image.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        # 构建每张图像的完整 3x3 颜色变换矩阵:
        #   M = P @ T @ Q   形状: (3, 3)
        # P: (3, k), T: (B, k, k), Q: (k, 3)
        # M: (B, 3, 3)
        P_exp = self.P.unsqueeze(0).expand(B, -1, -1)   # (B, 3, k)
        Q_exp = self.Q.unsqueeze(0).expand(B, -1, -1)   # (B, k, 3)
        M = torch.bmm(torch.bmm(P_exp, T), Q_exp)       # (B, 3, 3)

        # 应用映射: y = x @ M  => (B, H*W, 3)
        y = torch.bmm(x, M)

        # 还原形状: (B, H*W, 3) -> (B, 3, H, W)
        y = y.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

        return y

    def get_matrix(self, T: torch.Tensor) -> torch.Tensor:
        """返回完整的 3x3 颜色变换矩阵 M = P @ T @ Q。

        用于可视化或调试。

        Args:
            T: (B, k, k)

        Returns:
            M: (B, 3, 3)
        """
        B = T.shape[0]
        P_exp = self.P.unsqueeze(0).expand(B, -1, -1)
        Q_exp = self.Q.unsqueeze(0).expand(B, -1, -1)
        M = torch.bmm(torch.bmm(P_exp, T), Q_exp)
        return M
