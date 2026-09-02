"""分析和可视化 .cube LUT 文件。

用法:
  python analyze_lut.py --lut path/to/your.cube
  python analyze_lut.py --lut_dir data/luts        # 分析目录下所有 LUT

输出:
  1. 风格特征 (暖色/冷色、对比度、饱和度等)
  2. 可视化图像，展示颜色映射效果
  3. 在测试渐变图上的 before/after 对比
"""
import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.lut_utils import load_lut_from_cube, LUT3D


def analyze_lut_style(lut: LUT3D) -> dict:
    """从 LUT 数据分析风格特征。

    Args:
        lut: LUT3D 对象

    Returns:
        dict: 风格属性字典
    """
    data = lut.data  # (S, S, S, 3)
    S = lut.size

    # 恒等 LUT 用于对比
    identity = LUT3D._identity_lut(S)

    # 与恒等 LUT 的差异
    diff = data - identity  # (S, S, S, 3)

    # ---- 1. 色温 (暖色 vs 冷色) ----
    # 暖色: R 增加, B 减少
    # 冷色: R 减少, B 增加
    r_change = diff[..., 0].mean()
    b_change = diff[..., 2].mean()
    if r_change > 0.02 and b_change < -0.02:
        temp = "Warm (暖色)"
    elif r_change < -0.02 and b_change > 0.02:
        temp = "Cool (冷色)"
    elif abs(r_change) < 0.01 and abs(b_change) < 0.01:
        temp = "Neutral (中性)"
    else:
        temp = f"Subtle ({'warmish' if r_change > b_change else 'coolish'})"

    # ---- 2. 对比度 ----
    # 高对比度: 暗部更暗，亮部更亮
    # 检查对角线 (灰阶)
    diag_in = np.linspace(0, 1, S)
    diag_out = np.array([data[i, i, i, :] for i in range(S)])  # (S, 3)
    diag_gray_out = diag_out.mean(axis=1)

    # 线性拟合: 斜率 > 1 表示对比度更高
    slope = np.polyfit(diag_in, diag_gray_out, 1)[0]
    if slope > 1.1:
        contrast = f"High ({slope:.2f})"
    elif slope < 0.9:
        contrast = f"Low ({slope:.2f})"
    else:
        contrast = f"Normal ({slope:.2f})"

    # ---- 3. 饱和度 ----
    # 比较输出与输入的色彩丰富度
    # 采样一些饱和颜色
    sat_in = []
    sat_out = []
    for r, g, b in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
        ri = int(r * (S - 1))
        gi = int(g * (S - 1))
        bi = int(b * (S - 1))
        out_rgb = data[ri, gi, bi]
        sat_in.append(max(r, g, b) - min(r, g, b))
        sat_out.append(max(out_rgb) - min(out_rgb))
    sat_ratio = np.mean(sat_out) / max(np.mean(sat_in), 1e-6)
    if sat_ratio > 1.15:
        saturation = f"High ({sat_ratio:.2f}x)"
    elif sat_ratio < 0.85:
        saturation = f"Low ({sat_ratio:.2f}x)"
    else:
        saturation = f"Normal ({sat_ratio:.2f}x)"

    # ---- 4. 亮度 ----
    bright_change = diff.mean()
    if bright_change > 0.02:
        brightness = f"Brighter (+{bright_change:.3f})"
    elif bright_change < -0.02:
        brightness = f"Darker ({bright_change:.3f})"
    else:
        brightness = f"Normal ({bright_change:+.3f})"

    # ---- 5. 特殊: 黑白 / 单色 ----
    # 检查输出是否始终为灰色 (R==G==B)
    color_var = np.std(data[..., 0] - data[..., 1]) + np.std(data[..., 1] - data[..., 2])
    if color_var < 0.01:
        special = "Black & White (黑白)"
    else:
        special = "Color (彩色)"

    # ---- 6. 整体偏移 (色调) ----
    # 平均 RGB 偏移指示主导色调
    avg_shift = diff.reshape(-1, 3).mean(axis=0)
    tint_colors = ["R", "G", "B"]
    dominant = np.argmax(np.abs(avg_shift))
    if abs(avg_shift[dominant]) > 0.02:
        tint = f"{tint_colors[dominant]} shift ({avg_shift[dominant]:+.3f})"
    else:
        tint = "No tint"

    # ---- 7. 猜测风格名称 ----
    style_guess = []
    if "Warm" in temp:
        style_guess.append("暖色/日落")
    if "Cool" in temp:
        style_guess.append("冷色/冬季")
    if "High" in contrast:
        style_guess.append("高对比/电影感")
    if "Low" in saturation:
        style_guess.append("低饱和/复古")
    if "High" in saturation:
        style_guess.append("高饱和/鲜艳")
    if "B&W" in special:
        style_guess = ["黑白"]
    if not style_guess:
        style_guess = ["自然/微调"]

    return {
        "temperature": temp,
        "contrast": contrast,
        "saturation": saturation,
        "brightness": brightness,
        "color_type": special,
        "tint": tint,
        "style_guess": " + ".join(style_guess),
        "r_change": r_change,
        "b_change": b_change,
        "slope": slope,
        "sat_ratio": sat_ratio,
    }


def visualize_lut(lut: LUT3D, output_path: str, title: str = ""):
    """生成 LUT 的可视化图像。

    展示:
      - 色块 before/after
      - 灰阶 before/after
      - 原色/间色 before/after
    """
    S = lut.size
    data = lut.data

    fig_w, fig_h = 800, 600
    img = Image.new("RGB", (fig_w, fig_h), (30, 30, 30))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # 标题
    draw.text((10, 5), f"LUT Visualization: {title}", fill="white", font=font)

    # --- 第一部分: 色块 (before/after) ---
    y0 = 40
    cell = 40
    colors = [
        (0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
    ]

    draw.text((10, y0), "Original -> LUT applied:", fill="white", font=font_small)
    y0 += 20
    for i, (r, g, b) in enumerate(colors):
        ri = int(r * (S - 1))
        gi = int(g * (S - 1))
        bi = int(b * (S - 1))
        out = data[ri, gi, bi]

        x = 10 + i * (cell * 2 + 5)
        # 原始色块
        orig_rgb = (int(r * 255), int(g * 255), int(b * 255))
        draw.rectangle([x, y0, x + cell, y0 + cell], fill=orig_rgb)
        # 箭头
        draw.text((x + cell + 2, y0 + 5), "->", fill="white", font=font_small)
        # LUT 处理后色块
        out_rgb = (int(out[0] * 255), int(out[1] * 255), int(out[2] * 255))
        draw.rectangle([x + cell + 20, y0, x + cell * 2 + 20, y0 + cell], fill=out_rgb)

    # --- 第二部分: 灰阶 ---
    y0 = 120
    draw.text((10, y0), "Gray ramp (before/after):", fill="white", font=font_small)
    y0 += 20
    ramp_w = 60
    for i in range(S):
        val = i / (S - 1)
        ri = gi = bi = i
        out = data[ri, gi, bi]
        # 原始灰阶
        orig_rgb = int(val * 255)
        draw.rectangle([10 + i * ramp_w // S, y0, 10 + (i + 1) * ramp_w // S, y0 + 20],
                       fill=(orig_rgb, orig_rgb, orig_rgb))
        # LUT 处理后灰阶
        out_rgb = (int(out[0] * 255), int(out[1] * 255), int(out[2] * 255))
        draw.rectangle([10 + i * ramp_w // S, y0 + 25, 10 + (i + 1) * ramp_w // S, y0 + 45],
                       fill=out_rgb)

    # --- 第三部分: 完整色谱 ---
    y0 = 210
    draw.text((10, y0), "Color spectrum (before/after):", fill="white", font=font_small)
    y0 += 20
    spec_w = 780
    spec_h = 40
    for x in range(spec_w):
        r = (x / spec_w)
        g = 0.5
        b = 1 - (x / spec_w)
        ri = int(r * (S - 1))
        gi = int(g * (S - 1))
        bi = int(b * (S - 1))
        out = data[ri, gi, bi]
        # 原始色谱
        orig_rgb = (int(r * 255), int(g * 255), int(b * 255))
        draw.line([(10 + x, y0), (10 + x, y0 + spec_h)], fill=orig_rgb)
        # LUT 处理后色谱
        out_rgb = (int(out[0] * 255), int(out[1] * 255), int(out[2] * 255))
        draw.line([(10 + x, y0 + spec_h + 5), (10 + x, y0 + spec_h * 2 + 5)], fill=out_rgb)

    img.save(output_path)
    print(f"可视化图像已保存: {output_path}")


def apply_lut_to_image(lut: LUT3D, image_path: str, output_path: str):
    """将 LUT 应用到测试图像，保存 before/after 对比图。"""
    img = Image.open(image_path).convert("RGB")
    img = TF.resize(img, 256)
    tensor = TF.to_tensor(img).unsqueeze(0)
    out = lut.apply(tensor)
    out_img = TF.to_pil_image(out.squeeze(0).clamp(0, 1))

    # 左右拼接
    w, h = img.size
    combined = Image.new("RGB", (w * 2 + 10, h), (255, 255, 255))
    combined.paste(img, (0, 0))
    combined.paste(out_img, (w + 10, 0))
    combined.save(output_path)
    print(f"Before/After 对比图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析 LUT 风格")
    parser.add_argument("--lut", type=str, help=".cube 文件路径")
    parser.add_argument("--lut_dir", type=str, help="存放 .cube 文件的目录")
    parser.add_argument("--test_image", type=str, default=None,
                        help="可选的测试图像，用于 before/after 对比")
    parser.add_argument("--out_dir", type=str, default="lut_analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 收集要分析的 LUT 文件
    lut_files = []
    if args.lut:
        lut_files = [args.lut]
    elif args.lut_dir:
        lut_files = sorted(glob.glob(os.path.join(args.lut_dir, "*.cube")))
    else:
        print("请指定 --lut 或 --lut_dir")
        return

    print(f"找到 {len(lut_files)} 个 LUT 文件待分析\n")
    print("=" * 70)

    for lut_path in lut_files:
        name = os.path.splitext(os.path.basename(lut_path))[0]
        print(f"\n📄 {os.path.basename(lut_path)}")

        try:
            lut = load_lut_from_cube(lut_path)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        attrs = analyze_lut_style(lut)

        print(f"  风格猜测:  {attrs['style_guess']}")
        print(f"  色彩类型:  {attrs['color_type']}")
        print(f"  色温:      {attrs['temperature']}")
        print(f"  对比度:    {attrs['contrast']}")
        print(f"  饱和度:    {attrs['saturation']}")
        print(f"  亮度:      {attrs['brightness']}")
        print(f"  色调偏移:  {attrs['tint']}")

        # 保存可视化
        vis_path = os.path.join(args.out_dir, f"{name}_visualization.png")
        visualize_lut(lut, vis_path, title=name)

        # 如果提供了测试图像，生成对比图
        if args.test_image:
            cmp_path = os.path.join(args.out_dir, f"{name}_comparison.png")
            apply_lut_to_image(lut, args.test_image, cmp_path)

        print("-" * 70)


if __name__ == "__main__":
    main()
