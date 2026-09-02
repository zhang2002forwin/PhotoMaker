"""3D LUT 工具，用于训练时的颜色扰动。

支持功能:
  - 加载 .cube LUT 文件 (Adobe Cube 格式)
  - 通过三线性插值将 3D LUT 应用到图像
  - 随机 LUT 扰动 (无文件时自动生成随机平滑 LUT)
  - 随机滤镜调整 (亮度、对比度、饱和度、色相)

参考论文 (Sec. 4):
  "We use about 5,000 LUT files, along with the random image filter adjustment
   strategy [39], as input color perturbations during training."
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F


class LUT3D:
    """3D LUT 容器和应用器。

    Args:
        size: LUT 网格尺寸 (如 33)
        data: (size, size, size, 3) 的 numpy 数组，范围 [0, 1]
    """

    def __init__(self, size: int = 33, data: np.ndarray = None):
        self.size = size
        if data is not None:
            self.data = data
        else:
            # 恒等 LUT (输入=输出)
            self.data = self._identity_lut(size)

    @staticmethod
    def _identity_lut(size: int) -> np.ndarray:
        """生成恒等 3D LUT (输入颜色等于输出颜色)。"""
        idx = np.linspace(0, 1, size)
        r, g, b = np.meshgrid(idx, idx, idx, indexing="ij")
        lut = np.stack([r, g, b], axis=-1).astype(np.float32)
        return lut

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """通过三线性插值将 3D LUT 应用到一批图像。

        Args:
            image: (B, 3, H, W) 输入图像，范围 [0, 1]

        Returns:
            output: (B, 3, H, W) LUT 处理后的图像
        """
        device = image.device
        B, C, H, W = image.shape

        lut = torch.from_numpy(self.data).to(device)  # (S, S, S, 3)

        # 将像素值归一化到 LUT 索引范围 [0, S-1]
        S = self.size
        x = image.clamp(0, 1) * (S - 1)

        # 分离三个通道
        r = x[:, 0]
        g = x[:, 1]
        b = x[:, 2]

        # 三线性插值所需的下取整和上取整
        r0 = torch.floor(r).long().clamp(0, S - 2)
        g0 = torch.floor(g).long().clamp(0, S - 2)
        b0 = torch.floor(b).long().clamp(0, S - 2)
        r1 = r0 + 1
        g1 = g0 + 1
        b1 = b0 + 1

        # 小数部分: 形状 (B, 1, H, W, 1)，用于与 (B, H, W, 3) 广播
        dr = (r - r0.float()).unsqueeze(1).unsqueeze(-1)
        dg = (g - g0.float()).unsqueeze(1).unsqueeze(-1)
        db = (b - b0.float()).unsqueeze(1).unsqueeze(-1)

        # 收集 8 个角点的值
        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]

        # 三线性插值
        c00 = c000 * (1 - db) + c001 * db
        c01 = c010 * (1 - db) + c011 * db
        c10 = c100 * (1 - db) + c101 * db
        c11 = c110 * (1 - db) + c111 * db

        c0 = c00 * (1 - dg) + c01 * dg
        c1 = c10 * (1 - dg) + c11 * dg

        out = c0 * (1 - dr) + c1 * dr
        # out 形状: (B, 1, H, W, 3) -> squeeze -> (B, H, W, 3) -> (B, 3, H, W)
        out = out.squeeze(1)
        return out.permute(0, 3, 1, 2).contiguous()


def load_lut_from_cube(path: str, size: int = 33) -> LUT3D:
    """加载 .cube LUT 文件。

    Args:
        path: .cube 文件路径
        size: 预期 LUT 尺寸

    Returns:
        LUT3D 对象
    """
    data = np.zeros((size, size, size, 3), dtype=np.float32)
    with open(path, "r") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("TITLE"):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                # 索引映射: .cube 格式中 R 变化最快
                bi = idx // (size * size)
                gi = (idx // size) % size
                ri = idx % size
                data[ri, gi, bi] = [r, g, b]
                idx += 1

    return LUT3D(size=size, data=data)


def load_all_luts(lut_dir: str, size: int = 33) -> list:
    """从目录加载所有 .cube LUT 文件。

    Returns:
        LUT3D 对象列表
    """
    luts = []
    cube_files = glob.glob(os.path.join(lut_dir, "*.cube"))
    for path in cube_files:
        try:
            lut = load_lut_from_cube(path, size=size)
            luts.append(lut)
        except Exception as e:
            print(f"警告: 加载 {path} 失败: {e}")
    if len(luts) == 0:
        print(f"在 {lut_dir} 中未找到 .cube 文件，将使用随机 LUT。")
    else:
        print(f"从 {lut_dir} 加载了 {len(luts)} 个 LUT 文件")
    return luts


def generate_random_lut(size: int = 33, strength: float = 0.3) -> LUT3D:
    """生成随机平滑 3D LUT 用于颜色扰动。

    通过在粗网格上生成随机噪声再上采样，创建平滑的随机颜色映射。

    Args:
        size: LUT 网格尺寸
        strength: 扰动强度，范围 [0, 1]

    Returns:
        LUT3D 随机平滑映射
    """
    identity = LUT3D._identity_lut(size)
    # 在粗网格上生成随机噪声，然后上采样
    coarse_size = 4
    coarse_noise = np.random.uniform(
        -strength, strength, (coarse_size, coarse_size, coarse_size, 3)
    ).astype(np.float32)

    # 用 torch 的三线性插值上采样到完整 LUT 尺寸
    coarse_t = torch.from_numpy(coarse_noise).permute(3, 0, 1, 2).unsqueeze(0)
    full_t = F.interpolate(
        coarse_t, size=(size, size, size), mode="trilinear", align_corners=True
    )
    full_noise = full_t.squeeze(0).permute(1, 2, 3, 0).numpy()

    lut_data = np.clip(identity + full_noise, 0, 1)
    return LUT3D(size=size, data=lut_data)


def random_lut_perturbation(
    image: torch.Tensor,
    luts: list = None,
    size: int = 33,
    strength: float = 0.3,
    use_filter: bool = True,
) -> torch.Tensor:
    """对一批图像应用随机 LUT + 滤镜扰动。

    遵循论文做法: LUT 文件 + 随机图像滤镜调整。

    Args:
        image: (B, 3, H, W) 输入图像，范围 [0, 1]
        luts: 预加载的 LUT3D 对象列表。为 None 或空时生成随机 LUT。
        size: LUT 尺寸
        strength: 随机 LUT 的扰动强度
        use_filter: 是否同时应用随机滤镜调整

    Returns:
        perturbed: (B, 3, H, W) 扰动后的图像
    """
    B = image.shape[0]
    outputs = []

    for i in range(B):
        img = image[i:i+1]

        # 1. LUT 扰动
        if luts and len(luts) > 0:
            lut = luts[np.random.randint(len(luts))]
        else:
            lut = generate_random_lut(size=size, strength=strength)
        img = lut.apply(img)

        # 2. 随机滤镜调整 (亮度、对比度、饱和度、色相)
        if use_filter:
            img = random_filter_adjust(img)

        outputs.append(img)

    return torch.cat(outputs, dim=0)


def random_filter_adjust(image: torch.Tensor) -> torch.Tensor:
    """随机亮度 / 对比度 / 饱和度 / 色相调整。

    Args:
        image: (1, 3, H, W) 输入图像，范围 [0, 1]

    Returns:
        adjusted: (1, 3, H, W) 调整后的图像
    """
    # 亮度调整
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        image = image * factor

    # 对比度调整
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        mean = image.mean(dim=(2, 3), keepdim=True)
        image = (image - mean) * factor + mean

    # 饱和度调整
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        gray = image.mean(dim=1, keepdim=True)
        image = gray + (image - gray) * factor

    # 色相调整 (通过通道旋转近似)
    if np.random.rand() < 0.3:
        shift = np.random.uniform(-0.1, 0.1)
        image = torch.roll(image, shifts=int(shift * 10), dims=1)

    return image.clamp(0, 1)
