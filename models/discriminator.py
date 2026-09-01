"""风格判别器 D。

用于风格相似度度量 (论文 Appendix B) 和对抗训练。
输入一张图像，预测其所属的风格类别。

架构: 轻量级 CNN 分类器。
"""
import torch
import torch.nn as nn


class StyleDiscriminator(nn.Module):
    """风格判别器 / 分类器。

    预测输入图像的风格类别。
    同时用作对抗信号和风格相似度度量。

    Args:
        num_classes: 风格类别数
        feat_dim: 特征维度
    """

    def __init__(self, num_classes: int = 700, feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim

        # 特征提取网络: 4 层卷积下采样
        self.features = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_dim, feat_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_dim * 2, feat_dim * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_dim * 4, feat_dim * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 8, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, image: torch.Tensor):
        """对图像风格进行分类。

        Args:
            image: (B, 3, H, W)

        Returns:
            logits: (B, num_classes) 分类 logits
            feat:   (B, feat_dim*8) 特征嵌入
        """
        feat = self.features(image)
        feat = feat.flatten(1)
        logits = self.classifier(feat)
        return logits, feat

    def get_feature(self, image: torch.Tensor) -> torch.Tensor:
        """仅提取风格特征嵌入 (不分类)。"""
        feat = self.features(image)
        return feat.flatten(1)
