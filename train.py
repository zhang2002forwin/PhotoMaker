"""Neural Preset 训练脚本。

自监督训练流程:
  1. 加载 COCO 图像
  2. 通过 LUT + 滤镜生成颜色扰动对 (I_i, I_j)
  3. 训练编码器 E + DNCM:
       L_rec: 恒等重建 (输入 -> 输出应为恒等)
       L_con: 一致性 (I_i 和 I_j -> 相同 T, 相同输出)
       L_adv: 对抗损失 (可选)
  4. 可选训练判别器 D

用法:
  python train.py --coco_root /path/to/coco/train2017 --lut_dir /path/to/luts
"""
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_train_parser
from data.dataset import ColorPerturbationDataset
from models import NeuralPreset, StyleDiscriminator
from losses import (
    reconstruction_loss,
    consistency_loss,
    adversarial_loss,
    total_variation_loss,
)


def train(args):
    """主训练函数。"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载配置
    cfg = Config()
    cfg.BATCH_SIZE = args.batch_size
    cfg.NUM_EPOCHS = args.epochs
    cfg.LR_E = args.lr
    cfg.IMAGE_SIZE = args.image_size
    cfg.LR_STEP_SIZE = args.lr_step_size
    cfg.LR_GAMMA = args.lr_gamma
    cfg.CHECKPOINT_DIR = args.ckpt_dir
    cfg.COCO_ROOT = args.coco_root
    cfg.LUT_DIR = args.lut_dir

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # ==================== 数据 ====================
    dataset = ColorPerturbationDataset(
        coco_root=cfg.COCO_ROOT,
        lut_dir=cfg.LUT_DIR,
        image_size=cfg.IMAGE_SIZE,
        lut_size=cfg.LUT_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ==================== 模型 ====================
    model = NeuralPreset(k=cfg.K_DIM, thumb_size=cfg.THUMB_SIZE,
                         pretrained_path=args.pretrained_path).to(device)
    discriminator = StyleDiscriminator(
        num_classes=cfg.D_FEATURE_DIM, feat_dim=cfg.D_FEATURE_DIM
    ).to(device)

    # ==================== 优化器 ====================
    # 生成器 = 编码器 + DNCM
    opt_G = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.dncm.parameters()),
        lr=cfg.LR_E,
        betas=(0.9, 0.999),
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), lr=cfg.LR_D, betas=(0.9, 0.999)
    )

    # ==================== 学习率调度器 ====================
    # 论文: "The lr is multiplied by 0.1 after 24 epochs."
    scheduler_G = torch.optim.lr_scheduler.StepLR(
        opt_G, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
    )
    scheduler_D = torch.optim.lr_scheduler.StepLR(
        opt_D, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
    )

    # ==================== 恢复训练 ====================
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"从 epoch {start_epoch} 恢复训练")

    # ==================== TensorBoard ====================
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    # ==================== 训练循环 ====================
    step = 0
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        model.train()
        discriminator.train()

        epoch_losses = {"rec": 0, "con": 0, "adv_g": 0, "adv_d": 0, "total": 0}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            I = batch["I"].to(device)
            I_i = batch["I_i"].to(device)
            I_j = batch["I_j"].to(device)

            # ================================
            # 训练生成器 (编码器 E + DNCM)
            # ================================
            opt_G.zero_grad()

            # L_rec: 恒等重建
            # 当输入=目标 (无风格变化) 时，输出应等于输入
            out_identity, _ = model(I)
            loss_rec = reconstruction_loss(out_identity, I)

            # L_con: 两个扰动版本之间的一致性
            loss_con = consistency_loss(model, I_i, I_j)

            # L_adv: 对抗损失 (生成器部分)
            out_i, _ = model(I_i)
            loss_adv_g = adversarial_loss(
                discriminator, out_i, I_i, mode="generator"
            )

            # TV 正则化 (小权重)
            loss_tv = total_variation_loss(out_i)

            loss_G = (
                cfg.LAMBDA_REC * loss_rec
                + cfg.LAMBDA_CON * loss_con
                + cfg.LAMBDA_ADV * loss_adv_g
                + 0.001 * loss_tv
            )

            loss_G.backward()
            opt_G.step()

            # ================================
            # 训练判别器
            # ================================
            opt_D.zero_grad()
            loss_adv_d = adversarial_loss(
                discriminator, out_i.detach(), I_i, mode="discriminator"
            )
            loss_adv_d.backward()
            opt_D.step()

            # ---- 日志记录 ----
            epoch_losses["rec"] += loss_rec.item()
            epoch_losses["con"] += loss_con.item()
            epoch_losses["adv_g"] += loss_adv_g.item()
            epoch_losses["adv_d"] += loss_adv_d.item()
            epoch_losses["total"] += loss_G.item()
            n_batches += 1
            step += 1

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] "
                    f"Batch [{batch_idx}/{len(loader)}] "
                    f"L_rec={loss_rec.item():.4f} "
                    f"L_con={loss_con.item():.4f} "
                    f"L_adv_g={loss_adv_g.item():.4f} "
                    f"L_adv_d={loss_adv_d.item():.4f}"
                )
                writer.add_scalar("Loss/rec", loss_rec.item(), step)
                writer.add_scalar("Loss/con", loss_con.item(), step)
                writer.add_scalar("Loss/adv_g", loss_adv_g.item(), step)
                writer.add_scalar("Loss/adv_d", loss_adv_d.item(), step)
                writer.add_scalar("Loss/total_G", loss_G.item(), step)

        # ---- Epoch 汇总 ----
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        current_lr = opt_G.param_groups[0]["lr"]
        print(
            f"\n=== Epoch {epoch+1} 平均: "
            f"L_rec={avg['rec']:.4f} L_con={avg['con']:.4f} "
            f"L_adv_g={avg['adv_g']:.4f} L_adv_d={avg['adv_d']:.4f} "
            f"lr={current_lr:.2e} ===\n"
        )
        writer.add_scalar("LR/lr", current_lr, epoch)

        # ---- 学习率衰减 ----
        scheduler_G.step()
        scheduler_D.step()

        # ---- 保存检查点 ----
        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"neural_preset_epoch{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            },
            ckpt_path,
        )
        print(f"已保存检查点: {ckpt_path}")

        # 保存最新版本
        latest_path = os.path.join(cfg.CHECKPOINT_DIR, "latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            },
            latest_path,
        )

    writer.close()
    print("训练完成。")


if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()
    train(args)
