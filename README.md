# Neural Preset 复现

基于论文 *Neural Preset for Color Style Transfer* (CVPR 2023) 的复现实现。

论文官方仓库仅提供在线 Demo，未公开训练代码。本项目按论文思路完整复现了模型架构和自监督训练流程。

## 核心思路

### DNCM (Deterministic Neural Color Mapping)

论文的核心创新，用矩阵乘法替代 3D LUT：

```
Y = I(h*w, 3) @ P(3, k) @ T(k, k) @ Q(k, 3)
```

- `P`, `Q`：所有图像共享的可学习投影矩阵
- `T`：由编码器 E 从缩略图预测的图像自适应矩阵，仅 `k×k` 个参数（k=16 时仅 256 个）
- **确定性映射**：相同输入颜色 → 相同输出颜色，避免伪影
- **内存高效**：逐像素独立运算，支持 4K/8K

### 自监督训练

由于没有成对标注数据，论文采用自监督策略：

1. **数据来源**：MS COCO 图像 + ~5000 个 LUT 文件
2. **颜色扰动**：对每张图像 I 用 LUT + 随机滤镜生成两个扰动版本 I_i, I_j
3. **损失函数**：
   - `L_rec`：重建损失（扰动图 I_i 应能重建出原图 I，学习逆颜色变换）
   - `L_con`：一致性损失（同一图像的两个扰动应产生相同的 T 和输出）
   - `L_adv`：对抗损失（风格真实性）

## 项目结构

```
PhotoMaker/
├── config.py                 # 全局配置
├── requirements.txt          # 依赖
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── losses.py                 # 损失函数
├── models/
│   ├── __init__.py
│   ├── dncm.py               # DNCM 核心模块
│   ├── encoder.py            # 缩略图编码器 E
│   ├── discriminator.py      # 风格判别器 D
│   └── neural_preset.py      # 完整模型
└── data/
    ├── __init__.py
    ├── lut_utils.py          # 3D LUT 工具
    └── dataset.py            # 自监督数据集
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

### 1. MS COCO 数据集

下载 [MS COCO 2017 Train](http://images.cocodataset.org/zips/train2017.zip)，解压到：

```
data/coco/train2017/
```

### 2. LUT 文件（可选）

将 `.cube` 格式的 LUT 文件放入：

```
data/luts/
```

如果没有 LUT 文件，训练时会自动生成随机平滑 LUT 进行扰动。

## 训练

```bash
python train.py \
    --coco_root data/coco/train2017 \
    --lut_dir data/luts \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --image_size 256 \
    --device cuda
```

从检查点恢复训练：

```bash
python train.py --resume checkpoints/latest.pth --coco_root data/coco/train2017
```

使用 wandb 监控训练（默认启用，需先 `wandb login`）：

```bash
python train.py --wandb_project neural-preset ...
```

## 推理

### 风格迁移（内容图 + 风格参考图）

```bash
python inference.py \
    --mode transfer \
    --content content.jpg \
    --style style.jpg \
    --checkpoint checkpoints/latest.pth \
    --output output.jpg
```

### 提取并保存预设

```bash
python inference.py \
    --mode transfer \
    --content content.jpg \
    --style style.jpg \
    --checkpoint checkpoints/latest.pth \
    --save_preset my_preset.pt
```

### 应用已保存的预设

```bash
python inference.py \
    --mode preset \
    --input input.jpg \
    --preset my_preset.pt \
    --checkpoint checkpoints/latest.pth \
    --output output.jpg
```

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `K_DIM` | 16 | DNCM 瓶颈维度，T 矩阵为 16×16=256 个参数 |
| `THUMB_SIZE` | 256 | 编码器输入缩略图大小（论文固定为 256） |
| `ENCODER_BACKBONE` | efficientnet_b0 | 编码器骨干网络（论文使用 EfficientNet-B0） |
| `LUT_SIZE` | 33 | 3D LUT 网格大小 |
| `LAMBDA_REC` | 1.0 | 重建损失权重 |
| `LAMBDA_CON` | 1.0 | 一致性损失权重 |
| `LAMBDA_ADV` | 0.01 | 对抗损失权重 |

## 与论文的对应关系

| 论文内容 | 代码实现 |
|----------|----------|
| DNCM: Y = I·P·T·Q | `models/dncm.py` |
| 编码器 E 预测 T | `models/encoder.py` — EfficientNet-B0 骨干 + FC 头 |
| 判别器 D | `models/discriminator.py` |
| LUT 颜色扰动 | `data/lut_utils.py` |
| L_rec 重建损失 | `losses.py: reconstruction_loss()` |
| L_con 一致性损失 | `losses.py: consistency_loss()` |
| L_adv 对抗损失 | `losses.py: adversarial_loss()` |
| 自监督训练流程 | `train.py` |

## 注意事项

1. **LUT 文件**：论文使用约 5000 个 LUT 文件。如果没有，代码会生成随机平滑 LUT 替代，效果类似但可能略有差异。
2. **判别器 D**：论文中 D 主要用于风格相似度度量（Appendix B），需要 700+ 风格类别的标注数据。本实现用轻量分类器近似，对抗损失权重设为 0.01。
3. **训练数据**：论文使用 MS COCO，需自行下载。
4. **编码器初始化**：FC 最后一层权重零初始化、偏置初始化为展平的单位阵，使训练初期 T≈I_k；配合 DNCM 的 P@Q=I_3，初始映射 M≈I_3（输出≈输入）。
5. **EfficientNet-B0 预训练权重**：首次运行会自动从 torchvision 下载 ImageNet 预训练权重（约 20MB），缓存到 `~/.cache/torch/hub/checkpoints/`。

## 引用

```bibtex
@InProceedings{NeuralPreset,
  author = {Zhanghan Ke and Yuhao Liu and Lei Zhu and Nanxuan Zhao and Rynson W.H. Lau},
  title = {Neural Preset for Color Style Transfer},
  booktitle = {Computer Vision and Pattern Recognition Conference (CVPR)},
  year = {2023},
}
```
