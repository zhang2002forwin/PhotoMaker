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
import torchvision.transforms.functional as TF
from PIL import Image

import wandb

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


def run_validation(model, cfg, step, device):
    """在验证集上运行风格迁移并保存拼接结果。

    对 eval_data/content_img 中的每张内容图，
    与 eval_data/style_img 中的每张风格图做风格迁移，
    将 [content | style(压缩) | result] 横向拼接保存。

    两阶段流程:
      1. 从 style 图提取 {d_s, r_s}
      2. 从 content 图提取 {d_c, r_c}
      3. Z_c = nDNCM(content, d_c)
      4. result = sDNCM(Z_c, r_s)

    Args:
        model: NeuralPreset 模型
        cfg: 配置对象
        step: 当前全局 step
        device: 计算设备
    """
    content_dir = os.path.join(cfg.EVAL_DATA_DIR, "content_img")
    style_dir = os.path.join(cfg.EVAL_DATA_DIR, "style_img")
    out_dir = os.path.join(cfg.EVAL_RES_DIR, f"step_{step:07d}")
    os.makedirs(out_dir, exist_ok=True)

    content_files = sorted(
        f for f in os.listdir(content_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    style_files = sorted(
        f for f in os.listdir(style_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    model.eval()
    with torch.no_grad():
        for c_name in content_files:
            # 内容图: 保持原始大小 (不压缩)
            content = TF.to_tensor(Image.open(os.path.join(content_dir, c_name)).convert("RGB")).unsqueeze(0).to(device)
            for s_name in style_files:
                # 风格图: 压缩到 THUMB_SIZE
                style_img = Image.open(os.path.join(style_dir, s_name)).convert("RGB")
                style_img = TF.resize(style_img, cfg.THUMB_SIZE)
                style = TF.to_tensor(style_img).unsqueeze(0).to(device)

                # 从 style 图提取 {d_s, r_s}
                _, d_s, r_s = model(style)
                # 从 content 图提取 {d_c, r_c}
                Z_c, d_c, _ = model(content)
                # sDNCM: 风格化
                result = model.dncm(Z_c, r_s, use_nDNCM=False)

                # 拼接: content (原尺寸) | style (压缩) | result
                # content 保持原尺寸不压缩; style 保持压缩尺寸 (THUMB_SIZE); result 与 content 同尺寸
                H, W = content.shape[-2:]
                content_pil = TF.to_pil_image(content.squeeze(0).clamp(0, 1))
                # style 保持压缩尺寸 (THUMB_SIZE x THUMB_SIZE)，不放大
                style_pil = TF.to_pil_image(style.squeeze(0).clamp(0, 1))
                # result 与 content 同尺寸
                result_pil = TF.to_pil_image(result.squeeze(0).clamp(0, 1))
                result_pil = result_pil.resize((W, H), Image.LANCZOS)

                # 横向拼接 (canvas 高度取 content 高度，style 贴在其列顶部)
                total_w = content_pil.width + style_pil.width + result_pil.width
                canvas = Image.new("RGB", (total_w, H))
                x = 0
                for img in [content_pil, style_pil, result_pil]:
                    canvas.paste(img, (x, 0))
                    x += img.width

                out_name = f"{os.path.splitext(c_name)[0]}__{os.path.splitext(s_name)[0]}.jpg"
                canvas.save(os.path.join(out_dir, out_name), quality=95)

    model.train()
    print(f"[验证] step={step} 结果已保存到 {out_dir} ({len(content_files)*len(style_files)} 张)")
    return out_dir


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

    # ==================== wandb ====================
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "batch_size": cfg.BATCH_SIZE,
            "epochs": cfg.NUM_EPOCHS,
            "lr": cfg.LR_E,
            "lr_step_size": cfg.LR_STEP_SIZE,
            "lr_gamma": cfg.LR_GAMMA,
            "image_size": cfg.IMAGE_SIZE,
            "k_dim": cfg.K_DIM,
            "lambda_rec": cfg.LAMBDA_REC,
            "lambda_con": cfg.LAMBDA_CON,
            "lambda_adv": cfg.LAMBDA_ADV,
            "eval_interval": cfg.EVAL_INTERVAL,
            "pretrained_path": args.pretrained_path or "auto_download",
        },
    )
    os.makedirs(cfg.EVAL_RES_DIR, exist_ok=True)
    print(f"wandb 已启用: project={args.wandb_project}, run={wandb.run.name}")
    print(f"wandb 链接: {wandb.run.url}")

    # ==================== 训练循环 ====================
    step = 0
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        model.train()
        discriminator.train()

        epoch_losses = {"rec": 0, "con": 0, "adv_g": 0, "adv_d": 0, "total": 0}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            I_i = batch["I_i"].to(device)
            I_j = batch["I_j"].to(device)

            # ================================
            # 训练生成器 (编码器 E + DNCM)
            # ================================
            opt_G.zero_grad()

            # L_rec: 交叉重建损失 (论文 Eq. 6-7)
            #   Z_i = nDNCM(I_i, d_i),  Z_j = nDNCM(I_j, d_j)
            #   Y_i = sDNCM(Z_j, r_i),  Y_j = sDNCM(Z_i, r_j)
            # L_rec = ‖Y_i − I_i‖₁ + ‖Y_j − I_j‖₁
            loss_rec = reconstruction_loss(model, I_i, I_j)

            # L_con: 归一化空间一致性 (论文 Eq. 5)
            #   L_con = ‖Z_i − Z_j‖₂ = ‖nDNCM(I_i, d_i) − nDNCM(I_j, d_j)‖₂
            loss_con = consistency_loss(model, I_i, I_j)

            # L_adv: 对抗损失 (生成器部分)
            # 用 Y_i 作为生成结果
            Z_i, d_i, r_i = model(I_i)
            Y_i = model.dncm(Z_i, r_i, use_nDNCM=False)
            loss_adv_g = adversarial_loss(
                discriminator, Y_i, I_i, mode="generator"
            )

            # TV 正则化 (小权重)
            loss_tv = total_variation_loss(Y_i)

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
                discriminator, Y_i.detach(), I_i, mode="discriminator"
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

            # 定期验证和保存检查点
            if step % cfg.EVAL_INTERVAL == 0:
                run_validation(model, cfg, step, device)

                # 保存检查点
                ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"neural_preset_step{step:07d}.pth")
                torch.save(
                    {
                        "step": step,
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
                        "step": step,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "opt_G": opt_G.state_dict(),
                        "opt_D": opt_D.state_dict(),
                    },
                    latest_path,
                )

            # 每个 step 记录 loss 到 wandb (step 级曲线)
            wandb.log({
                "Step/rec": loss_rec.item(),
                "Step/con": loss_con.item(),
                "Step/adv_g": loss_adv_g.item(),
                "Step/adv_d": loss_adv_d.item(),
                "Step/total_G": loss_G.item(),
                "Step/epoch": epoch + 1,
            }, step=step)

            # 每 50 个 batch 打印一次
            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] "
                    f"Batch [{batch_idx}/{len(loader)}] "
                    f"L_rec={loss_rec.item():.4f} "
                    f"L_con={loss_con.item():.4f} "
                    f"L_adv_g={loss_adv_g.item():.4f} "
                    f"L_adv_d={loss_adv_d.item():.4f}"
                )

        # ---- Epoch 汇总 ----
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        current_lr = opt_G.param_groups[0]["lr"]
        print(
            f"\n=== Epoch {epoch+1} 平均: "
            f"L_rec={avg['rec']:.4f} L_con={avg['con']:.4f} "
            f"L_adv_g={avg['adv_g']:.4f} L_adv_d={avg['adv_d']:.4f} "
            f"lr={current_lr:.2e} ===\n"
        )
        wandb.log({
            "Epoch/rec": avg["rec"],
            "Epoch/con": avg["con"],
            "Epoch/adv_g": avg["adv_g"],
            "Epoch/adv_d": avg["adv_d"],
            "Epoch/total": avg["total"],
            "LR/lr": current_lr,
        }, step=epoch)

        # ---- 学习率衰减 ----
        scheduler_G.step()
        scheduler_D.step()

    wandb.finish()
    print("训练完成。")


if __name__ == "__main__":
    parser = get_train_parser()
    args = parser.parse_args()
    train(args)
