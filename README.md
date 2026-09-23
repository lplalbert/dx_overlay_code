# DX Overlay Code

水印叠加与检测系统。将数字 ID 经 RS(15,5) 编码后以棋盘格网格嵌入屏幕截图，支持从截图中检测并还原 ID。

## 系统架构

```
dx_overlay_code/
├── C++ 水印叠加引擎          main.cpp, overlay*.cpp/h
├── 水印编码生成              generate_yellow_white_template.py
│                             rs_gen_Syn_template_nums_dual.py
│                             wm_generator_slim.py
├── 噪声模型                  physical_moire.py, wechat_worst_case_compressor.py
└── watermark_locator/        检测模型 (v1 独立 / v2 联合 × vv1 U-Net / vv2 YOLOv8)
    ├── dataset/              合成数据集生成
    ├── generate_locator_pattern.py
    ├── utils.py
    ├── v1/                   独立识别
    │   ├── vv1_unet/         U-Net 语义分割
    │   └── vv2_yolov8/       YOLOv8 目标检测
    └── v2/                   联合识别
        ├── vv1_unet/
        └── vv2_yolov8/
```

## 编码方案

- **RS(15,5)** 纠错编码, GF(2^4)
- **棋盘格网格**: 4行×6列 = 24 块, 其中 6 块为定位块
- **斜条纹调制**: 45° 方向, 周期 4px, 宽度 2px — 仅作用于"黄色"(信号)单元格
- **双嵌入域** (`channel_mode`):
  - `b` — RGB B 通道 (黄/白, 对应 `generate_yellow_white_template.py`)
  - `cb` — YCbCr Cb/Cr 通道 (对应 HLSL shader)

## 快速开始

### 生成合成数据集

```bash
cd watermark_locator/dataset

# B 通道嵌入
python generate_dataset.py --num_samples 1000 --output_dir ./data \
    --channel_mode b --carrier_path /path/to/carrier.png

# Cb 通道嵌入
python generate_dataset.py --num_samples 1000 --output_dir ./data \
    --channel_mode cb --carrier_dir /path/to/carriers/
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
| `tile_crop` | 3×3 平铺+随机裁剪 | warpAffine |
| `pimog` | 透视(±30px)+光照+摩尔纹+高斯 | warpPerspective |

## 依赖

```bash
pip install -r requirements.txt
pip install ultralytics  # YOLOv8 训练需要
```

U-Net 模型代码位于 `watermark_locator/v1/vv1_unet/unet/` (基于 [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet))。

## 标注格式

**vv1** (U-Net 分割): PNG 灰度 mask, 255=定位块区域

**vv2** (YOLO 检测): 每行 `class cx cy w h` (归一化到 0-1)
