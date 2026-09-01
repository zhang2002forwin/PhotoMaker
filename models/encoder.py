"""缩略图编码器 E。

从输入图像的小缩略图预测图像自适应变换矩阵 T (k*k)。
使用 EfficientNet-B0 作为骨干网络。

参考论文 (Sec. Implementation, Page 7):
  "We adopt EfficientNet-B0 [67] as the encoder E in Neural Preset.
   We fix the input size of E to 256x256."

  - 输入: 256x256 的缩略图
  - 骨干: EfficientNet-B0 (ImageNet 预训练)
  - 输出: T 矩阵，形状 (k, k)，展平后预测再 reshape
  - 每张图像仅约 256 个自适应参数 (k=16)
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class ThumbnailEncoder(nn.Module):
    """基于 EfficientNet-B0 的编码器，从缩略图预测 T (k*k)。

    Args:
        thumb_size: 输入缩略图空间尺寸 (论文使用 256)
        k: 输出 T 矩阵为 k*k
        pretrained: 是否使用 ImageNet 预训练权重
        in_channels: 输入通道数 (3=RGB，4=RGB+mask 用于和谐化任务)
    """

    def __init__(
        self,
        thumb_size: int = 256,
        k: int = 16,
        pretrained: bool = True,
        pretrained_path: str = None,
        in_channels: int = 3,
    ):
        super().__init__()
        self.k = k
        self.thumb_size = thumb_size
        self.out_dim = k * k
        self.in_channels = in_channels

        # 加载 EfficientNet-B0 骨干网络
        if pretrained_path:
            # 从本地路径加载预训练权重
            self.backbone = tvm.efficientnet_b0(weights=None)
            state_dict = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"已从本地加载 EfficientNet-B0 权重: {pretrained_path}")
        elif pretrained:
            try:
                weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
                self.backbone = tvm.efficientnet_b0(weights=weights)
            except Exception:
                # 兼容旧版 torchvision
                self.backbone = tvm.efficientnet_b0(pretrained=True)
        else:
            self.backbone = tvm.efficientnet_b0(weights=None)

        # 如果输入通道数不为 3 (例如 RGB+mask 用于和谐化)，
        # 替换第一层卷积
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            # 用旧权重初始化前 3 个通道，额外通道零初始化
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
            self.backbone.features[0][0] = new_conv

        # EfficientNet-B0 特征维度 = 1280
        feat_dim = self.backbone.classifier[1].in_features  # 1280

        # 移除分类头，只保留特征提取器
        self.backbone.classifier = nn.Identity()

        # FC 头，预测 T (展平的 k*k)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.out_dim),
        )

        # 将 FC 最后一层零初始化，使训练初期 T≈0，映射接近恒等 (无颜色变化)
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, thumbnail: torch.Tensor) -> torch.Tensor:
        """从缩略图预测 T 矩阵。

        Args:
            thumbnail: (B, C, S, S) 输入缩略图，范围 [0, 1]

        Returns:
            T: (B, k, k) 预测的变换矩阵
        """
        feat = self.backbone(thumbnail)      # (B, 1280)
        T_flat = self.fc(feat)               # (B, k*k)
        T = T_flat.view(-1, self.k, self.k)  # (B, k, k)
        return T
