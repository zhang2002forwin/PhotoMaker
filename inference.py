"""Neural Preset 推理脚本。

两种模式:
  1. 风格迁移: 给定内容图 + 风格参考图，从风格图提取 T，应用到内容图。
  2. 应用预设: 加载已保存的 T 矩阵 (预设)，应用到图像/视频。

用法:
  # 风格迁移
  python inference.py --mode transfer --content content.jpg --style style.jpg --checkpoint checkpoints/latest.pth

  # 应用预设
  python inference.py --mode preset --input input.jpg --preset preset.pt --checkpoint checkpoints/latest.pth
"""
import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models import NeuralPreset


def load_image(path: str, size: int = None) -> torch.Tensor:
    """加载图像为 (1, 3, H, W) 张量，范围 [0, 1]。"""
    img = Image.open(path).convert("RGB")
    if size:
        img = TF.resize(img, size)
    return TF.to_tensor(img).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str):
    """将 (1, 3, H, W) 张量保存为图像。"""
    img = tensor.squeeze(0).clamp(0, 1)
    img = TF.to_pil_image(img)
    img.save(path)


def extract_preset(model: NeuralPreset, style_image: torch.Tensor) -> torch.Tensor:
    """从风格参考图提取 T (预设)。

    Args:
        model: NeuralPreset 模型
        style_image: (1, 3, H, W) 风格参考图

    Returns:
        T: (1, k, k) 预设矩阵
    """
    model.eval()
    with torch.no_grad():
        import torch.nn.functional as F
        thumb = F.adaptive_avg_pool2d(style_image, model.thumb_size)
        T = model.encoder(thumb)
    return T


def apply_preset(model: NeuralPreset, image: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """将预设 T 应用到图像。

    Args:
        model: NeuralPreset 模型
        image: (1, 3, H, W) 输入图像
        T: (1, k, k) 预设矩阵

    Returns:
        output: (1, 3, H, W) 输出图像
    """
    model.eval()
    with torch.no_grad():
        output = model.dncm(image, T)
    return output


def main():
    parser = argparse.ArgumentParser(description="Neural Preset 推理")
    parser.add_argument("--mode", type=str, default="transfer",
                        choices=["transfer", "preset"],
                        help="transfer: 内容图+风格图; preset: 应用已保存的 T")
    parser.add_argument("--content", type=str, help="内容图路径")
    parser.add_argument("--style", type=str, help="风格参考图路径")
    parser.add_argument("--input", type=str, help="输入图路径 (preset 模式)")
    parser.add_argument("--preset", type=str, help="已保存预设 .pt 文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="EfficientNet-B0 预训练权重本地路径 (不填则自动下载)")
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_preset", type=str, default=None,
                        help="将提取的 T 保存到此路径")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # 加载模型
    model = NeuralPreset(k=cfg.K_DIM, thumb_size=cfg.THUMB_SIZE,
                         pretrained_path=args.pretrained_path).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载检查点: {args.checkpoint}")

    if args.mode == "transfer":
        # 风格迁移: 从风格图提取 T，应用到内容图
        content = load_image(args.content).to(device)
        style = load_image(args.style).to(device)

        T = extract_preset(model, style)
        print(f"提取的预设 T 形状: {T.shape}")

        if args.save_preset:
            torch.save(T, args.save_preset)
            print(f"预设已保存到 {args.save_preset}")

        output = apply_preset(model, content, T)
        save_image(output, args.output)
        print(f"输出已保存到 {args.output}")

    elif args.mode == "preset":
        # 应用已保存的预设
        input_img = load_image(args.input).to(device)
        T = torch.load(args.preset, map_location=device)
        output = apply_preset(model, input_img, T)
        save_image(output, args.output)
        print(f"输出已保存到 {args.output}")


if __name__ == "__main__":
    main()
