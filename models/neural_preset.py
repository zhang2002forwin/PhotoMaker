"""Neural Preset 模型。

两阶段颜色迁移管道 (论文 Sec. 3.2):
  1. nDNCM (颜色归一化): 将输入图像映射到归一化颜色空间
  2. sDNCM (颜色风格化): 将归一化图像映射到目标风格

编码器 E 从缩略图预测 {d, r} 参数:
  - d: nDNCM 参数 (颜色归一化)
  - r: sDNCM 参数 (颜色风格化)

训练时采用交叉重建 (论文 Sec. 3.3, Eq. 6-7):
  Z_i = nDNCM(I_i, d_i),  Z_j = nDNCM(I_j, d_j)
  Y_i = sDNCM(Z_j, r_i),  Y_j = sDNCM(Z_i, r_j)
"""
import torch
import torch.nn as nn

from .dncm import DNCM
from .encoder import ThumbnailEncoder


class NeuralPreset(nn.Module):
    """两阶段 Neural Preset 模型。

    Args:
        k: 瓶颈维度，T 矩阵为 k*k
        thumb_size: 缩略图输入尺寸 (论文使用 256)
        pretrained_path: 预训练 EfficientNet-B0 路径 (可选)
    """

    def __init__(
        self,
        k: int = 16,
        thumb_size: int = 256,
        pretrained_path: str = None,
    ):
        super().__init__()
        self.k = k
        self.thumb_size = thumb_size

        # 编码器 E: 从缩略图预测 {d, r}
        self.encoder = ThumbnailEncoder(
            thumb_size=thumb_size,
            k=k,
            pretrained=True,
            pretrained_path=pretrained_path,
        )

        # nDNCM (颜色归一化) + sDNCM (颜色风格化)
        # 两者共享同一个 DNCM 模块，使用不同的 P/Q 投影矩阵
        self.dncm = DNCM(k=k)

    def forward(self, image: torch.Tensor):
        """前向传播。

        论文 Eq. 3: Z = nDNCM(I, d), 其中 {d, r} = E(Ĩ)

        Args:
            image: (B, 3, H, W) 输入图像，范围 [0, 1]

        Returns:
            Z: (B, 3, H, W) 归一化后的图像
            d: (B, k, k) nDNCM 参数
            r: (B, k, k) sDNCM 参数
        """
        # 缩略图
        thumb = torch.nn.functional.adaptive_avg_pool2d(
            image, (self.thumb_size, self.thumb_size)
        )
        # 编码器预测 {d, r}
        d, r = self.encoder(thumb)
        # nDNCM: 颜色归一化
        Z = self.dncm(image, d, use_nDNCM=True)
        return Z, d, r

    def stylize(self, image: torch.Tensor, d: torch.Tensor, r: torch.Tensor):
        """两阶段风格化: nDNCM → sDNCM。

        论文 Eq. 3-4:
          Z = nDNCM(I, d)
          Y = sDNCM(Z, r)

        Args:
            image: (B, 3, H, W) 输入图像
            d: (B, k, k) nDNCM 参数
            r: (B, k, k) sDNCM 参数

        Returns:
            Y: (B, 3, H, W) 风格化后的图像
        """
        Z = self.dncm(image, d, use_nDNCM=True)
        Y = self.dncm(Z, r, use_nDNCM=False)
        return Y
