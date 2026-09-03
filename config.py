"""Neural Preset 训练和推理的全局配置。

本文件定义了模型结构、训练超参数、数据路径等所有可配置项。
修改这里的参数即可调整训练行为，无需改动其他代码。
"""
import argparse


class Config:
    """全局配置类。

    所有默认值均参考论文 *Neural Preset for Color Style Transfer* (CVPR 2023)。
    """

    # ==================== DNCM 配置 ====================
    # 核心公式: Y = I(h*w,3) @ P(3,k) @ T(k,k) @ Q(k,3)
    # k 是瓶颈维度，k=16 时 T 矩阵仅有 256 个参数
    K_DIM = 16

    # ==================== 编码器 E 配置 ====================
    # 论文原文: "We fix the input size of E to 256x256."
    THUMB_SIZE = 256
    # 编码器骨干网络: EfficientNet-B0 (ImageNet 预训练)
    ENCODER_BACKBONE = "efficientnet_b0"
    # EfficientNet-B0 预训练权重本地路径 (None 则自动从 torchvision 下载)
    PRETRAINED_PATH = None
    # 编码器输出维度 = K_DIM * K_DIM
    ENCODER_OUT_DIM = K_DIM * K_DIM  # 256

    # ==================== 判别器 D 配置 ====================
    # 特征维度，用于风格分类和对抗训练
    D_FEATURE_DIM = 128

    # ==================== 训练超参数 ====================
    # 论文 Section 4.1: "We use Adam optimizer with initial lr=3e-4,
    #                   batch size=24, λ=10. The lr is multiplied by 0.1
    #                   after 24 epochs."
    BATCH_SIZE = 24
    LR_E = 3e-4          # 编码器 + DNCM 参数 (P, Q) 的学习率
    LR_D = 3e-4          # 判别器的学习率
    NUM_EPOCHS = 100
    IMAGE_SIZE = 256     # 训练图像尺寸
    NUM_WORKERS = 4      # 数据加载的并行进程数

    # 学习率调度: 每 24 个 epoch 衰减为 0.1 倍
    LR_STEP_SIZE = 2
    LR_GAMMA = 0.1

    # 损失函数权重 (参考论文 λ=10)
    LAMBDA_REC = 1.0      # 重建损失权重
    LAMBDA_CON = 10.0     # 一致性损失权重 (论文 λ=10)
    LAMBDA_ADV = 0.00    # 对抗损失权重 (设较小值，参考色彩迁移常用做法)

    # ==================== LUT 扰动配置 ====================
    LUT_SIZE = 33         # 3D LUT 网格尺寸 (33 级是常用尺寸)
    LUT_DIR = "data/luts" # 存放 .cube LUT 文件的目录

    # ==================== 数据集路径 ====================
    COCO_ROOT = "data/coco/train2017"  # MS COCO 训练集目录
    VAL_IMAGES = "data/val_images"     # 验证图像目录

    # ==================== 验证集配置 ====================
    EVAL_DATA_DIR = "eval_data"       # 验证集目录 (含 content_img/ 和 style_img/)
    EVAL_RES_DIR = "eval_res"         # 验证结果保存目录
    EVAL_INTERVAL = 200               # 每多少 step 做一次验证

    # ==================== 输出路径 ====================
    CHECKPOINT_DIR = "checkpoints"  # 模型检查点保存目录

    # ==================== 设备 ====================
    DEVICE = "cuda"


def get_train_parser():
    """创建训练脚本的命令行参数解析器。

    所有参数都有默认值，也可以通过命令行覆盖。
    """
    p = argparse.ArgumentParser(description="Train Neural Preset")
    p.add_argument("--coco_root", type=str, default=Config.COCO_ROOT,
                   help="MS COCO train2017 目录路径")
    p.add_argument("--lut_dir", type=str, default=Config.LUT_DIR,
                   help="存放 .cube LUT 文件的目录路径")
    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=Config.LR_E)
    p.add_argument("--image_size", type=int, default=Config.IMAGE_SIZE)
    p.add_argument("--lr_step_size", type=int, default=Config.LR_STEP_SIZE,
                   help="每 N 个 epoch 学习率衰减")
    p.add_argument("--lr_gamma", type=float, default=Config.LR_GAMMA,
                   help="学习率衰减系数")
    p.add_argument("--device", type=str, default=Config.DEVICE)
    p.add_argument("--pretrained_path", type=str, default=Config.PRETRAINED_PATH,
                   help="EfficientNet-B0 预训练权重本地路径 (不填则自动下载)")
    p.add_argument("--ckpt_dir", type=str, default=Config.CHECKPOINT_DIR,
                   help="模型检查点保存目录 (默认 checkpoints)")
    p.add_argument("--resume", type=str, default=None,
                   help="从指定检查点恢复训练")
    # ==================== wandb 参数 ====================
    p.add_argument("--wandb_project", type=str, default="neural-preset",
                   help="wandb 项目名称")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="wandb run 名称 (不填则自动生成)")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="wandb entity/团队名称 (个人账号可不填)")
    return p
