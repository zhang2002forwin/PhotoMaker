"""Neural Preset 自监督训练数据集。

参考论文 (Sec. 4):
  - 源图像: MS COCO
  - 颜色扰动: ~5000 个 LUT 文件 + 随机滤镜调整
  - 对每张图像 I，生成两个扰动版本 I_i, I_j
  - 训练样本对:
      (I_i, I_j) 用于一致性损失 L_con
      (I, I)    恒等映射用于重建损失 L_rec
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .lut_utils import random_lut_perturbation, load_all_luts


class ColorPerturbationDataset(Dataset):
    """从 COCO 生成颜色扰动图像对的数据集。

    Args:
        coco_root: COCO 图像目录路径 (如 train2017)
        lut_dir:   存放 .cube LUT 文件的目录
        image_size: 训练图像尺寸
        lut_size:  LUT 网格尺寸
    """

    def __init__(
        self,
        coco_root: str,
        lut_dir: str = None,
        image_size: int = 256,
        lut_size: int = 33,
    ):
        self.image_size = image_size
        self.lut_size = lut_size

        # 收集图像路径
        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.image_paths.extend(glob.glob(os.path.join(coco_root, ext)))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {coco_root} 中未找到图像")

        # 加载 LUT
        self.luts = []
        if lut_dir and os.path.isdir(lut_dir):
            self.luts = load_all_luts(lut_dir, size=lut_size)

        print(f"数据集: {len(self.image_paths)} 张图像, {len(self.luts)} 个 LUT")

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        """加载并预处理图像，返回 (1, 3, H, W)，范围 [0, 1]。"""
        img = Image.open(path).convert("RGB")
        # 调整大小
        img = TF.resize(img, self.image_size)
        # 中心裁剪为正方形
        w, h = img.size
        if w != h:
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            img = TF.crop(img, top, left, min_side, min_side)
            img = TF.resize(img, self.image_size)
        img = TF.to_tensor(img)  # (3, H, W) 范围 [0, 1]
        return img.unsqueeze(0)

    def __getitem__(self, idx: int):
        """返回一个训练样本。

        Returns:
            dict 包含:
              I:      原始图像 (3, H, W)
              I_i:    扰动版本 1 (3, H, W)
              I_j:    扰动版本 2 (3, H, W)
        """
        path = self.image_paths[idx]
        I = self._load_image(path)  # (1, 3, H, W)

        # 生成两个扰动版本
        I_i = random_lut_perturbation(
            I, luts=self.luts, size=self.lut_size, use_filter=True
        )
        I_j = random_lut_perturbation(
            I, luts=self.luts, size=self.lut_size, use_filter=True
        )

        return {
            "I": I.squeeze(0),
            "I_i": I_i.squeeze(0),
            "I_j": I_j.squeeze(0),
        }
