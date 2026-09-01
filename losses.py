"""Neural Preset 训练用损失函数。

参考论文，总损失为:
  L = LAMBDA_REC * L_rec + LAMBDA_CON * L_con + LAMBDA_ADV * L_adv

  - L_rec:  重建损失 (恒等映射应保持不变)
            当输入和目标风格相同时，输出应等于输入。
  - L_con:  一致性损失 (相同内容 -> 相同预设)
            给定同一图像的两个扰动版本 I_i 和 I_j，
            模型预测的 T_i 和 T_j 应产生一致的结果。
  - L_adv:  对抗损失 (可选，用于提升风格真实感)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_rec: L1 重建损失。

    模型在恒等映射场景下 (输入即目标) 应能重建出原始图像。

    Args:
        output: (B, 3, H, W) 模型输出
        target: (B, 3, H, W) 目标图像

    Returns:
        loss: 标量损失值
    """
    return F.l1_loss(output, target)


def consistency_loss(
    model: nn.Module,
    I_i: torch.Tensor,
    I_j: torch.Tensor,
) -> torch.Tensor:
    """L_con: 一致性损失。

    论文核心思想: 同一图像的两个扰动版本，经过模型去除扰动后，
    应该产生相同的"预设" (T 矩阵) 和相同的输出。

    约束条件:
      1. T_i ≈ T_j  (预测的变换矩阵应一致)
      2. model(I_i) ≈ model(I_j)  (输出应一致)

    Args:
        model: NeuralPreset 模型
        I_i: (B, 3, H, W) 扰动版本 1
        I_j: (B, 3, H, W) 扰动版本 2

    Returns:
        loss: 标量损失值
    """
    out_i, T_i = model(I_i)
    out_j, T_j = model(I_j)

    # T 矩阵一致性
    loss_T = F.l1_loss(T_i, T_j)

    # 输出一致性
    loss_out = F.l1_loss(out_i, out_j)

    return loss_T + loss_out


def adversarial_loss(
    discriminator: nn.Module,
    output: torch.Tensor,
    real: torch.Tensor,
    mode: str = "generator",
) -> torch.Tensor:
    """L_adv: 对抗损失 (LSGAN 风格，最小二乘 GAN)。

    Args:
        discriminator: 风格判别器 StyleDiscriminator
        output: (B, 3, H, W) 生成图像
        real:   (B, 3, H, W) 真实风格图像
        mode:   "generator" (训练生成器) 或 "discriminator" (训练判别器)

    Returns:
        loss: 标量损失值
    """
    if mode == "generator":
        # 生成器希望判别器将输出判定为真 (标签为 1)
        logits, _ = discriminator(output)
        target = torch.ones_like(logits[:, 0])
        return F.mse_loss(logits[:, 0], target)
    else:
        # 判别器需要区分真伪
        logits_real, _ = discriminator(real)
        logits_fake, _ = discriminator(output.detach())
        target_real = torch.ones_like(logits_real[:, 0])
        target_fake = torch.zeros_like(logits_fake[:, 0])
        loss_real = F.mse_loss(logits_real[:, 0], target_real)
        loss_fake = F.mse_loss(logits_fake[:, 0], target_fake)
        return 0.5 * (loss_real + loss_fake)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """TV 损失，用于空间平滑正则化 (可选)。

    鼓励相邻像素差异较小，减少噪声伪影。
    """
    B, C, H, W = image.shape
    tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], 2).mean()
    tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], 2).mean()
    return tv_h + tv_w
