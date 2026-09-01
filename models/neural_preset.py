"""NeuralPreset: 完整模型，组合编码器 E + DNCM。

处理流程:
  1. 将输入图像 I 下采样为缩略图
  2. 编码器 E 从缩略图预测 T 矩阵
  3. DNCM 应用 Y = I @ P @ T @ Q 生成颜色映射后的输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dncm import DNCM
from .encoder import ThumbnailEncoder


class NeuralPreset(nn.Module):
    """完整的 Neural Preset 模型: 编码器 + DNCM。

    Args:
        k: DNCM 瓶颈维度
        thumb_size: 编码器输入缩略图尺寸
    """

    def __init__(self, k: int = 16, thumb_size: int = 256, pretrained_path: str = None):
        super().__init__()
        self.k = k
        self.thumb_size = thumb_size

        self.encoder = ThumbnailEncoder(thumb_size=thumb_size, k=k, pretrained_path=pretrained_path)
        self.dncm = DNCM(k=k)

    def forward(self, image: torch.Tensor):
        """完整前向传播。

        Args:
            image: (B, 3, H, W) 输入图像，范围 [0, 1]

        Returns:
            output: (B, 3, H, W) 颜色映射后的图像
            T:      (B, k, k) 预测的变换矩阵
        """
        # 通过平均池化生成缩略图
        thumbnail = F.adaptive_avg_pool2d(image, self.thumb_size)

        # 编码器从缩略图预测 T
        T = self.encoder(thumbnail)

        # DNCM 应用颜色映射
        output = self.dncm(image, T)

        return output, T

    def apply_preset(self, image: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """将预计算的 T (预设) 应用到图像。

        这就是"预设"概念: T 是预设，可跨图像/视频复用。
        """
        return self.dncm(image, T)
