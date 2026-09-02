"""Neural Preset 训练用损失函数。

参考论文，总损失为:
  L = LAMBDA_REC * L_rec + LAMBDA_CON * L_con + LAMBDA_ADV * L_adv

论文损失 (Sec. 3.3):
  - L_con (Eq. 5): 一致性损失，约束归一化空间 Z
    L_con = ‖nDNCM(I_i, d_i) − nDNCM(I_j, d_j)‖₂
  - L_rec (Eq. 7): 重建损失，交叉重建
    L_rec = ‖Y_i − I_i‖₁ + ‖Y_j − I_j‖₁
    其中 Y_i = sDNCM(Z_j, r_i), Y_j = sDNCM(Z_i, r_j)
  - L_adv: 对抗损失 (可选)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(
    model: nn.Module,
    I_i: torch.Tensor,
    I_j: torch.Tensor,
) -> torch.Tensor:
    """L_rec: 交叉重建损失 (论文 Eq. 7)。

    两阶段交叉重建 (论文 Eq. 6):
      Z_i = nDNCM(I_i, d_i),  Z_j = nDNCM(I_j, d_j)
      Y_i = sDNCM(Z_j, r_i),  Y_j = sDNCM(Z_i, r_j)
    L_rec = ‖Y_i − I_i‖₁ + ‖Y_j − I_j‖₁

    Args:
        model: NeuralPreset 模型
        I_i: (B, 3, H, W) 扰动版本 1
        I_j: (B, 3, H, W) 扰动版本 2

    Returns:
        loss: 标量损失值
    """
    # 两阶段前向
    Z_i, d_i, r_i = model(I_i)
    Z_j, d_j, r_j = model(I_j)

    # 交叉重建: Y_i = sDNCM(Z_j, r_i), Y_j = sDNCM(Z_i, r_j)
    Y_i = model.dncm(Z_j, r_i, use_nDNCM=False)
    Y_j = model.dncm(Z_i, r_j, use_nDNCM=False)

    # L1 重建损失
    loss = F.l1_loss(Y_i, I_i) + F.l1_loss(Y_j, I_j)
    return loss


def consistency_loss(
    model: nn.Module,
    I_i: torch.Tensor,
    I_j: torch.Tensor,
) -> torch.Tensor:
    """L_con: 一致性损失 (论文 Eq. 5)。

    约束归一化颜色空间 Z 的一致性:
      L_con = ‖Z_i − Z_j‖₂
      = ‖nDNCM(I_i, d_i) − nDNCM(I_j, d_j)‖₂

    同一图像的两个扰动版本，经过 nDNCM 归一化后应得到相同结果。

    Args:
        model: NeuralPreset 模型
        I_i: (B, 3, H, W) 扰动版本 1
        I_j: (B, 3, H, W) 扰动版本 2

    Returns:
        loss: 标量损失值
    """
    # nDNCM: 颜色归一化
    Z_i, _, _ = model(I_i)
    Z_j, _, _ = model(I_j)

    # L2 一致性损失
    return F.mse_loss(Z_i, Z_j)


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
        logits, _ = discriminator(output)
        target = torch.ones_like(logits[:, 0])
        return F.mse_loss(logits[:, 0], target)
    else:
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
