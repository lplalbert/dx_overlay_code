# DX Overlay Code

水印叠加与检测系统。将数字 ID 经 RS(15,5) 编码后以棋盘格网格嵌入屏幕截图，支持从截图中检测并还原 ID。

## 系统架构

```
dx_overlay_code/
├── C++ 水印叠加引擎          main.cpp, overlay*.cpp/h
├── 水印编码生成              generate_yellow_white_template.py
│                             rs_gen_Syn_template_nums_dual.py
├── 噪声模型                  physical_moire.py, wechat_worst_case_compressor.py
├── tools/decode/             匹配滤波解码器 + 判据实验 (见其 README)
├── docs/                     定位块检测方案说明 + 绘图脚本
├── vis/                      可视化输出
└── watermark_locator/        检测模型 (v1 独立识别 × vv1 U-Net / vv2 YOLOv8)
    ├── dataset/              合成数据集生成
    ├── generate_locator_pattern.py
    ├── utils.py
    ├── v1/                   独立识别
    │   ├── vv1_unet/         U-Net 语义分割
    │   └── vv2_yolov8/       YOLOv8 目标检测
    └── v2/                   第二代 (lattice + RT-DETR/YOLOv10/v12 对比)
```

## 编码方案

- **RS(15,5)** 纠错编码, GF(2^4)
- **块网格**: 4行×6列 = 24 块, 其中 6 块为定位块 (回字形同心方环)
- **斜条纹调制**: 45° 方向, 周期 4px, 宽度 2px — 仅作用于"黄色"(信号)单元格
- **嵌入域** (`channel_mode`): `b` / `yw` — 黄/白 RGB (对应 `generate_yellow_white_template.py`)
  - 模板**只含 0 和 255**: 黄(信号)=(B,G,R)=(0,255,255), 白(中性)=(255,255,255)
  - 按 α 融合: `out = α_eff·template + (1-α_eff)·carrier`, `α_eff = α·dynamicMask`
  - 白像素 `dynamicMask=0` 完全不改; 黄像素 `max|Δ| = α×255` (α=0.032 → 8.16)

## 快速开始

### 生成合成数据集

```bash
cd watermark_locator/dataset

# 多数据集 clean/noisy 双树 (推荐)
python prepare_multids.py --data_root /data1/lpl/datasets \
    --output_root /data1/lpl/datasets_labeled \
    --dataset_counts coco_minator_dataset=2000,document_ds=2000,bcgd=0 \
    --alpha 0.032 --channel_mode b --stage both --wechat_preset mainstream_worst

# 单批生成
python generate_dataset.py --num_samples 1000 --output_dir ./data \
    --channel_mode b --carrier_path /path/to/carrier.png
```

### 训练 (U-Net)

```bash
cd watermark_locator/v1/vv1_unet
python train.py --config config.yaml --device 0
```

### 训练 (YOLOv8)

```bash
cd watermark_locator/v1/vv2_yolov8
python train.py --config config.yaml --device 0
```

### 生成黄/白模板

```bash
python generate_yellow_white_template.py --id 123456 \
    --width 1920 --height 1080 --output templates/
```

## 噪声模型

数据集生成时随机施加 pair 噪声 (从以下 4 种中选 2 种组合):

| 噪声 | 几何变换 | 标签同步 |
|---|---|---|
| `identity` | 无 | — |
| `wechat` | 无 (JPEG 压缩) | 不变 |
| `tile_crop` | 3×3 平铺+随机旋转(±5°)+随机裁剪 | warpAffine |
| `pimog` | 透视(3%–7%)+畸变+光照+摩尔纹 | warp |

## 工具与解码

`tools/decode/` 是**匹配滤波解码器** —— 检测器不在环, 纯 CPU, 用来按**解码率**
(而不是残余) 卡档位上界、噪声仿真参数和实拍链上界。跑之前必须过自检:

```bash
python tools/decode/decode_cross_tier.py --selftest
# 自检通过 8/8      8 个已知 wm_id 的灰底合成图必须解回同一个 id
# 幻觉通道已封 3/3  纯平/纯噪声必须被拒, 不能幻觉出 id=0
```

其余脚本 (实拍链的极性/色偏/包络/多帧 A/B、按实拍召回选权重、RS 接口探针) 见
[tools/decode/README.md](tools/decode/README.md)。

定位块两种检测方案的说明与绘图脚本见
[docs/locator_detection.md](docs/locator_detection.md)。

## 依赖

```bash
pip install -r requirements.txt
pip install ultralytics  # YOLOv8 训练需要
```

U-Net 模型代码位于 `watermark_locator/v1/vv1_unet/unet/` (基于 [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet))。

## 标注格式

**vv1** (U-Net 分割): PNG 灰度 mask, 255=定位块区域

**vv2** (YOLO 检测): 每行 `class cx cy w h` (归一化到 0-1)
