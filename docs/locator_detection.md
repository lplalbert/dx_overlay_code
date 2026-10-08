# 定位块检测算法说明

> 本文档说明水印系统中**定位块（locator block）**的两种检测方案：**vv1 U-Net 语义分割** 与 **vv2 YOLOv8 目标检测**，
> 并给出训练曲线、验证集指标与测试图检测效果。
>
> 训练仍在进行中，文中指标为截至 **2026-09-24 15:00** 的快照；曲线图与效果图可由 `docs/make_train_curves.py`、
> `docs/make_detection_vis.py` 重新生成。

---

## 1. 任务背景

水印系统把数字 ID 嵌入屏幕截图，靠**定位块 + 解码器**还原 ID。流程是：

```
载体截图 ──► 拼 1920×1080 ──► 叠水印 (α=0.032) ──► [拍照 / 微信压缩] ──► 检测定位块 ──► 校正网格 ──► RS(15,5) 解码
                                                                 ↑
                                                          本文档讲这一步
```

**定位块的作用**是给整幅水印提供几何锚点。画面经过拍照透视、摩尔纹、JPEG 压缩后，
消息块的位置会漂移；只要 6 个定位块被检出，就能拟合出仿射/透视变换，把网格拉回原位再解码。

### 1.1 屏幕与块网格

| 项 | 值 |
|---|---|
| 屏幕分辨率 | `1920 × 1080` |
| 块网格 | `4 行 × 6 列` = 24 块 |
| 单块尺寸 | `320 × 270` |
| 消息区（块内嵌入区） | `160 × 135`（块的右下四分之一） |
| 块类型 | `k = (j + (i%2)*2) % 4`；`k==3` 为定位块 |
| 定位块数量 | **6** 个 |
| 定位块网格坐标 `(行, 列)` | `(0,3) (1,1) (1,5) (2,3) (3,1) (3,5)` |
| 定位块绝对矩形 `(x, y, w, h)` | `(j*320+160, i*270+135, 160, 135)` |

6 个定位块在画面上呈**交错棋盘状**分布（避开四角、避开正中），
这样任意局部遮挡或裁切后，仍能在剩余区域拟合出几何变换。

### 1.2 定位图案：回字形

定位块内部是 **8×8 同心方环（回字形）**，不是伪随机图案：

```
Y Y Y Y Y Y Y Y        Y = -1  黄色(信号)   条纹调制, 水印写在这里
Y W W W W W W Y        W = +1  白色(中性)   始终不变
Y W Y Y Y Y W Y
Y W Y W W Y W Y        ring 0 最外框  信号  28 格
Y W Y W W Y W Y        ring 1 间隙    中性  20 格
Y W Y Y Y Y W Y        ring 2 内框    信号  12 格
Y W W W W W W Y        ring 3 核心    中性   4 格  (2×2)
Y Y Y Y Y Y Y Y        ─────────────────────
                       信号 40 / 中性 24
```

选回字形的理由：**强角点 + 独特自相关峰**。同心方环在自相关面上有清晰的十字峰，
角点在梯度图里响应极强，CNN 和模板匹配都容易锚定；伪随机图案在重压缩后高频分量损失严重，
而回字形的环带是低频结构，JPEG 之后轮廓依然完整。

### 1.3 水印嵌入

- **融合公式** `out = carrier·(1−a_eff) + wm·a_eff`，`a_eff = α · dynamic_mask`
- **模板取值** 严格二值：信号 `0`、中性 `255`（BGR 黄 = `(0,255,255)`，白 = `(255,255,255)`，G/R 恒 255）
- **最大像素改动** `α × 255 = 0.032 × 255 = 8.16`（只落在 B 通道）
- **条纹调制** 45° 对角条纹（周期 4、带宽 2），只有约 25.4% 的像素实际被改写
- **训练输入** 仍是完整 **3 通道 BGR**（ch0 = B = 水印通道），不是 Cb 单通道

### 1.4 三种噪声

数据集为每个干净样本派生 3 个噪声版本，噪声函数严格复用既有实现：

| 标记 | 说明 | 调用链 |
|---|---|---|
| `wechat` | 单独微信压缩 | `add_wechat_noise(image)` ← `wechat_worst_case_compressor.py` preset `mainstream_worst` |
| `pimog` | 单独模拟拍照 | `add_pimog_noise(image, mask, bboxes, rng)` ← `physical_moire.py` preset `screen_capture`（EXTREME 档） |
| `pimog_wechat` | 拍照后微信压缩 | `add_pimog_noise(...)` → `add_wechat_noise(...)` |

**pimog 几何形变**：峰值位移在 **0%–5% 短边**（1080 → 0–54 px）内均匀采样，
残差场经 `displacement = residual × (target / residual_peak)` 归一化，峰值位移数学上严格等于采样值。
透视参数 `PERSPECTIVE_RANGE = (0.03, 0.07)` 只决定残差场的**形状**，不决定幅度。
含几何噪声时 GT 包围框会随形变一起更新，标签不丢准。

---

## 2. 两种检测算法

两个模型解决的是同一个问题——**在 1920×1080 噪声截图上框出 6 个定位块**——
但走的是两条技术路线。

| | **vv1 U-Net** | **vv2 YOLOv8** |
|---|---|---|
| 范式 | 语义分割（像素级） | 目标检测（框级） |
| 骨干 | U-Net（4 层下采样 + 双线性上采样） | YOLOv8n（CSP + PAN-FPN） |
| 输入尺寸 | `960 × 540` | `640`（letterbox） |
| 输出 | `1×540×960` 概率图 | 若干 `(cls, cx, cy, w, h, conf)` |
| 损失 | `BCE + Dice` | `box + cls + DFL` |
| 优化器 | `AdamW` lr=5e-4, wd=1e-4 | `SGD` lr=5e-3 |
| 调度 | `CosineAnnealingLR` | 内置 warmup + 余弦 |
| 后处理 | `sigmoid > 0.5` → 连通域 → 外接框 | `conf ≥ 0.25` + NMS |
| 优势 | 像素级边界、对弱纹理稳健 | 快、框回归准、直接出置信度 |

### 2.1 vv1 — U-Net 语义分割

**网络**（`unet/unet_model.py`）：标准 4 级 U-Net，`n_channels=3, n_classes=1, bilinear=True`。

```
输入 BGR 3×540×960
  ├─ DoubleConv  →  64
  ├─ Down ×4     →  128 / 256 / 512 / 512 (bilinear 时瓶颈 1024//2)
  ├─ Up ×4       →  512 / 256 / 128 / 64   （带 skip connection）
  └─ OutConv     →  1×540×960 (logits)
```

**训练**：像素级二分类，损失 = `BCEWithLogits + DiceLoss`。Dice 对前景稀疏（定位块只占画面约 2%）更敏感，
能把模型从"全预测背景"里拉出来。验证指标为像素级 Dice / IoU / P / R（阈值 0.5）。

**推理与后处理**（`docs/make_detection_vis.py:run_vv1 / mask_to_boxes`）：

1. 原图 resize 到 `960×540`，归一化到 `[0,1]`，CHW 送入网络
2. `sigmoid > 0.5` 得到二值掩码，`INTER_NEAREST` 放大回 `1920×1080`
3. `MORPH_CLOSE 5×5` 闭运算补小孔
4. `connectedComponentsWithStats` 取连通域，面积 ≥ 200 px 且宽高 > 4 的算一个候选
5. 每个连通域的外接矩形即为预测框

分割路线的好处是**边界贴合**：即便定位块被部分遮挡或与背景纹理粘连，
像素级响应仍能圈出残存的环带结构，再由连通域还原成框。

### 2.2 vv2 — YOLOv8 目标检测

**网络**：`yolov8n.pt` 单类别（`cls='0'`），输入 `imgsz=640` letterbox。

**标签格式** YOLO 归一化 `class cx cy w h`，例如：

```
0 0.625790 0.185171 0.084182 0.126962
0 0.292515 0.437504 0.082957 0.124680
```

每行一个定位块，一图固定 6 行（含几何噪声时坐标随形变更新）。

**训练**：`SGD` + 显式 `lr0`（ultralytics 的 `optimizer=auto` 会静默忽略 `lr0`，故显式指定 `SGD`），
`box_loss + cls_loss + DFL`，`patience=30` 早停。验证即标准 COCO 式 P / R / mAP@50 / mAP@50-95。

**推理**（`run_vv2`）：`model.predict(source=img_bgr, imgsz=640, conf=0.25, device=...)`，
NMS 后直接得到带置信度的框，**不需要后处理**。这是它比 vv1 快的主要原因。

### 2.3 评价口径

框级指标用 **IoU ≥ 0.5 的贪心匹配**：

- **tp** 正确检出——预测框与某个 GT 框 IoU ≥ 0.5 且该 GT 未被占用
- **fp** 误检/多报——预测框没匹配上任何 GT
- **fn** 漏检——GT 框没被任何预测框匹配上

```
Precision = tp / (tp + fp)     查准：报出来的有多少是对的
Recall    = tp / (tp + fn)     查全：该检出的检到了多少
F1        = 2PR / (P + R)
```

每图 GT 恒为 6，所以 `tp + fn = 6`。

---

## 3. 训练配置

### 3.1 数据集

三个来源数据集拼 1920×1080（**原生分辨率裁剪/平铺，不 resize**）：

| 来源 | train | val | 说明 |
|---|---|---|---|
| `coco_minator_dataset` | 1600 | 400 | 自然图像纹理 |
| `document_ds` | 1600 | 400 | 文档版面 |
| `bcgd` | 183 | 46 | 复杂背景 |
| **合计（clean）** | **3383** | **846** | |
| **合计（noisy ×3）** | **10149** | **2538** | 每张 clean 派生 3 个噪声版本 |

clean 树与噪声树**同 stem 配对**（`{clean_stem}_{wechat|pimog|pimog_wechat}`），
几何噪声的 mask 与 bbox 一并更新。

### 3.2 两阶段 + 续训

| 阶段 | 数据 | 轮数 | 说明 |
|---|---|---|---|
| clean 预训练 | `clean` | 10 | 先学干净图上的回字形结构 |
| noisy 微调 | `noisy`（pair 噪声） | 50 | 学抗噪 |
| **3-noise 续训**（本次） | `noisy`（3 噪声） | 50 | 在上一步 best 权重上继续，覆盖更广的噪声分布 |

本次续训分别从 v1 的 `best_model_noisy_finetune.pth` / `yolo_noisy_finetune/weights/best.pt` 起步，
配置见 `watermark_locator/v1/vv1_unet/config_v2.yaml` 与 `vv2_yolov8/config_v2.yaml`。
数据侧 0%–5% 的几何形变相当于**持续的数据增广**，是本次指标显著优于上一轮的重要原因。

---

## 4. 训练曲线

曲线由 `docs/make_train_curves.py` 从 `train_vv1_v2.log` 与 `results.csv` 直接解析生成，
**每条曲线单独一图**，存放于 `docs/figures/curves/`。

### 4.1 vv1 U-Net（截至 epoch 9 / 50，仍在训练）

| Epoch | Train Loss | Val Dice | Val IoU | Val P | Val R |
|---|---|---|---|---|---|
| 1 | 0.0659 | 0.9711 | 0.9445 | 0.9816 | 0.9615 |
| 2 | 0.0291 | 0.9818 | 0.9646 | 0.9866 | 0.9775 |
| 3 | 0.0201 | 0.9800 | 0.9611 | 0.9869 | 0.9735 |
| 4 | 0.0156 | 0.9851 | 0.9709 | 0.9908 | 0.9798 |
| 5 | 0.0133 | 0.9860 | 0.9726 | 0.9895 | 0.9827 |
| 6 | 0.0111 | 0.9889 | 0.9782 | 0.9907 | 0.9873 |
| 7 | 0.0093 | 0.9886 | 0.9776 | 0.9944 | 0.9830 |
| 8 | 0.0086 | 0.9883 | 0.9771 | 0.9931 | 0.9838 |
| **9** | **0.0077** | **0.9905** | **0.9812** | 0.9911 | **0.9900** |

- Train Loss 从 0.066 单调降到 0.008，未见反弹
- Val Dice 从 0.971 稳步爬到 **0.9905**，ep3 有一次小幅回撤后继续上行
- 当前 best 为 **ep9，Dice = 0.9905**（远高于上一轮 0.9309）

![vv1 Train Loss](figures/curves/vv1_train_loss.png)
![vv1 Val Dice](figures/curves/vv1_val_dice.png)
![vv1 Val IoU](figures/curves/vv1_val_iou.png)
![vv1 Val Precision](figures/curves/vv1_val_precision.png)
![vv1 Val Recall](figures/curves/vv1_val_recall.png)

### 4.2 vv2 YOLOv8（50 / 50 已完成）

关键节点：

| Epoch | box_loss | cls_loss | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| 1 | 0.373 | 0.337 | 0.973 | 0.970 | 0.992 | 0.965 |
| 15 | 0.319 | 0.245 | 0.991 | 0.989 | 0.995 | 0.978 |
| 30 | 0.264 | 0.194 | 0.995 | 0.992 | 0.995 | 0.987 |
| 43 | 0.204 | 0.151 | 0.997 | 0.996 | 0.995 | **0.990** |
| **50** | **0.178** | **0.129** | **0.9986** | **0.9984** | **0.995** | **0.9904** |

- 三项损失全程单调下降，50 轮末仍在缓降（`box_loss` 0.373 → 0.178）
- **Precision 0.9986 / Recall 0.9984** —— 两者接近饱和且互相咬合，说明既不误报也不漏检
- **mAP@50-95 = 0.9904**（ep49 峰值 0.9914），框回归非常准
- lr 由 5e-3 余弦退火至 1.5e-4，末段收敛平滑

![vv2 Box Loss](figures/curves/vv2_box_loss.png)
![vv2 Cls Loss](figures/curves/vv2_cls_loss.png)
![vv2 DFL Loss](figures/curves/vv2_dfl_loss.png)
![vv2 Val Precision](figures/curves/vv2_val_precision.png)
![vv2 Val Recall](figures/curves/vv2_val_recall.png)
![vv2 mAP@50](figures/curves/vv2_val_map50.png)
![vv2 mAP@50-95](figures/curves/vv2_val_map5095.png)

---

## 5. 验证集指标（框级，IoU ≥ 0.5）

在 **2538 张验证噪声图**（`coco_minator_dataset` 1200 + `document_ds` 1200 + `bcgd` 138）上做贪心匹配统计。
原始结果存于 `vis/report/val_metrics.json`。

共 **2538 张**验证噪声图（每图 6 个 GT 框 → 共 **15228** 个 GT），当前 best 权重：

### 5.1 总体（micro）

| 模型 | tp | fp | fn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **vv1 U-Net** | 15171 | **646** | 57 | 0.9592 | 0.9963 | 0.9774 |
| **vv2 YOLOv8** | 15210 | **84** | 18 | **0.9945** | **0.9988** | **0.9967** |

两个模型**查全率都在 99.6% 以上**（57 / 18 个漏检），差异主要在**误报**：
vv1 多报 646 个框，vv2 只多报 84 个，因此 vv2 的 P/F1 更高。

### 5.2 分数据集

| 数据集 | 图数 | GT | 模型 | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|---|---|---|
| `coco_minator_dataset` | 1200 | 7200 | vv1 | 0.9464 | 0.9974 | 0.9712 | 7181 / 407 / 19 |
| | | | vv2 | **0.9917** | **0.9988** | **0.9952** | 7191 / 60 / 9 |
| `document_ds` | 1200 | 7200 | vv1 | 0.9710 | 0.9954 | 0.9831 | 7167 / 214 / 33 |
| | | | vv2 | **0.9969** | **0.9988** | **0.9978** | 7191 / 22 / 9 |
| `bcgd` | 138 | 828 | vv1 | 0.9705 | 0.9940 | 0.9821 | 823 / 25 / 5 |
| | | | vv2 | **0.9976** | **1.0000** | **0.9988** | 828 / 2 / 0 |

vv2 在三类载体上全面领先；`bcgd`（复杂背景）两个模型都最好，
`coco_minator_dataset`（自然图像纹理复杂）是相对最难的一类 —— vv1 的 407 个误报大多落在这里。

### 5.3 训练日志内的像素级指标（vv1）/ 框级指标（vv2）

上面是**两模型同一框级口径**的对齐比较。各自训练时的原生指标：

| | 口径 | 指标 | 当前值 |
|---|---|---|---|
| vv1 | 像素级（阈值 0.5） | Dice / IoU / P / R | 0.9905 / 0.9812 / 0.9911 / 0.9900（ep9） |
| vv2 | 框级（COCO 式） | P / R / mAP@50 / mAP@50-95 | 0.9986 / 0.9984 / 0.9950 / 0.9904（ep50） |

---

## 6. 测试图检测效果

测试集 `/data1/lpl/datasets/test` 是 3 张真实 Snipaste 桌面截图（**不在训练分布内**）：

| 编号 | 文件 | 亮度均值 | 特征 |
|---|---|---|---|
| 01 | `Snipaste_2026-09-24_08-21-10.png` | 45.0 | 暗色桌面 |
| 02 | `Snipaste_2026-09-24_08-21-53.png` | 118.3 | 图标桌面 |
| 03 | `Snipaste_2026-09-24_08-23-04.png` | 222.1 | 亮色屏幕 |

每张图叠加水印后分别加 3 种噪声（`wechat` / `pimog` / `pimog_wechat`），
每种噪声输出 **4 张独立图**（不拼接）：

1. **clean + GT** —— 干净水印图，绿框为真值定位块
2. **noisy + GT** —— 噪声图，绿框为随几何形变更新后的真值
3. **vv1 检测** —— 品红框为 U-Net 分割后处理出的预测框
4. **vv2 检测** —— 品红框为 YOLOv8 预测框（带置信度）

图存于 `vis/report/detection/`，下采样副本（宽 1280）存于 `docs/figures/detection/`。

### 6.1 逐图结果（IoU ≥ 0.5，格式 `tp/fp/fn`）

| 样本 | 载体特征 | 噪声 | GT | vv1 | vv2 | vv2 置信度 |
|---|---|---|---|---|---|---|
| **01** | 暗色桌面（45.0） | wechat | 6 | 6/0/0 | 6/0/0 | 0.957–0.970 |
| | | pimog | 6 | 6/0/0 | 6/1/0 | 0.849–0.950 |
| | | pimog_wechat | 6 | 6/0/0 | 6/0/0 | 0.869–0.964 |
| **02** | 图标桌面（118.3） | wechat | 6 | 4/9/2 | 0/1/6 | 0.295 |
| | | pimog | 6 | 5/2/1 | 0/0/6 | — |
| | | pimog_wechat | 6 | 3/8/3 | 1/3/5 | 0.287–0.438 |
| **03** | 亮色屏幕（222.1） | wechat | 6 | 5/0/1 | 6/0/0 | 0.906–0.977 |
| | | pimog | 6 | 5/2/1 | 6/0/0 | 0.823–0.968 |
| | | pimog_wechat | 6 | 5/1/1 | 6/0/0 | 0.657–0.931 |
| **合计** | 9 图 / 54 框 | | 54 | **45/22/9** | **37/5/17** | |

样本 01、03 上两个模型几乎全检出；**样本 02（密集图标桌面）是共同的失败案例**。

### 6.2 成功案例 — 样本 01 暗色桌面（拍照后微信压缩）

vv1 与 vv2 均 6/0/0 全检出：

**① clean + GT**（绿 = 真值定位框）

![01 clean GT](figures/detection/01_Snipaste_2026-09-24_08-21-10_clean.png)

**② noisy + GT**（经 pimog 几何形变后 GT 已随之更新）

![01 noisy GT](figures/detection/01_Snipaste_2026-09-24_08-21-10_pimog_wechat_noisy.png)

**③ vv1 U-Net 检测**（品红 = 预测框）

![01 vv1](figures/detection/01_Snipaste_2026-09-24_08-21-10_pimog_wechat_vv1.png)

**④ vv2 YOLOv8 检测**（品红 = 预测框 + 置信度）

![01 vv2](figures/detection/01_Snipaste_2026-09-24_08-21-10_pimog_wechat_vv2.png)

### 6.3 成功案例 — 样本 03 亮色屏幕（拍照后微信压缩）

vv1 5/1/1，vv2 6/0/0：

![03 noisy GT](figures/detection/03_Snipaste_2026-09-24_08-23-04_pimog_wechat_noisy.png)
![03 vv1](figures/detection/03_Snipaste_2026-09-24_08-23-04_pimog_wechat_vv1.png)
![03 vv2](figures/detection/03_Snipaste_2026-09-24_08-23-04_pimog_wechat_vv2.png)

### 6.4 失败案例 — 样本 02 图标桌面（拍照后微信压缩）

vv1 3/8/3，vv2 1/3/5：

![02 noisy GT](figures/detection/02_Snipaste_2026-09-24_08-21-53_pimog_wechat_noisy.png)
![02 vv1](figures/detection/02_Snipaste_2026-09-24_08-21-53_pimog_wechat_vv1.png)
![02 vv2](figures/detection/02_Snipaste_2026-09-24_08-21-53_pimog_wechat_vv2.png)

**失败原因**：桌面布满高对比度应用图标，图标外框的**直角边缘**与回字形定位块的强角点响应高度相似。

- **vv1** 把图标误当成定位块 → 多报（fp=8）。U-Net 是像素级响应，图标边框的梯度结构同样激活了前景通道。
- **vv2** 置信度掉到 0.29–0.44，大量框被 `conf ≥ 0.25` 阈值卡掉或 NMS 合并 → 漏检（fn=5）。
  YOLO 的分类分支在"图标密集"这一未见分布上不确定，直接体现为低置信度。

样本 02 的载体是真实桌面截图，图标密度远超训练集中的 `coco_minator_dataset` 拼块纹理，
属于**训练分布外**。这说明当前模型对"高频直角结构"的判别性还不够强。

完整 9 组 × 4 张独立图存于 `vis/report/detection/`（下采样副本见 `docs/figures/detection/`）。

---

## 7. 结论

**在验证集（分布内，2538 张）上**：

| | vv1 U-Net | vv2 YOLOv8 |
|---|---|---|
| F1 | 0.9774 | **0.9967** |
| Precision | 0.9592 | **0.9945** |
| Recall | 0.9963 | **0.9988** |
| 误报数 | 646 | **84** |

**vv2 综合更优**：误报少 7.7 倍，漏检少 3 倍，且直接输出框与置信度、推理更快。
vv1 的差距几乎全部来自误报（646 vs 84）——像素级连通域对纹理复杂的 `coco_minator_dataset` 偏敏感。

**在测试图（分布外真实桌面，9 图）上**，结论会翻转：

| | vv1 U-Net | vv2 YOLOv8 |
|---|---|---|
| P / R / F1 | 0.672 / **0.833** / 0.744 | **0.881** / 0.685 / **0.771** |
| 样本 01、03（8 图中 6 图） | 几乎满分 | **满分** |
| 样本 02（密集图标） | 3/8/3（不漏但误报） | 1/3/5（**几乎全漏**） |

**分工建议**：

1. **优先用 vv2** —— 分布内表现、推理速度、置信度可解释性都更好。
   高置信度阈值（如 `conf ≥ 0.6`）可把误报压到极低，代价是密集图标场景漏检。
2. **vv1 作为召回兜底** —— 像素级响应在图标干扰下仍能找到部分定位块，
   适合在 vv2 检出数 < 6 时补漏；代价是需要额外的假阳性抑制（可结合回字形模板校验过滤）。
3. **两路融合**（推荐的最终方案）：vv2 高置信框 ∪ vv1 候选框，再用回字形自相关/模板匹配做一致性校验，
   既压误报又保召回。定位块位置在 4×6 网格中是固定的 6 个，
   检出后还可以按网格先验剔除明显不合理的位置。

**当前的薄弱环节**是**密集小图标 + 高对比度直角边缘**这类未见分布载体。
训练数据（`coco_minator_dataset` 自然图像 / `document_ds` 文档 / `bcgd` 复杂背景）
没有覆盖"满屏图标"的桌面纹理。要补这个短板，最直接的做法是把真实桌面截图纳入训练集，
或在合成阶段加入图标样式的高频角点纹理作负样本/干扰项。

（注：vv1 仍在训练至 50 轮，上述为 epoch 9 快照；vv2 已完成 50 轮。）

---

## 附录：复现

```bash
# 训练曲线（纯 CPU，秒级）
python docs/make_train_curves.py

# 检测效果图 + 验证集评测（需 GPU）
CUDA_VISIBLE_DEVICES=2 python docs/make_detection_vis.py
```

相关代码：

| 文件 | 作用 |
|---|---|
| `watermark_locator/generate_locator_pattern.py` | 回字形定位图案构造 |
| `watermark_locator/dataset/generate_dataset.py` | 水印叠加、块网格、噪声接口 |
| `watermark_locator/dataset/prepare_noise_from_clean.py` | 复用 clean 并行生成 3 噪声树 |
| `physical_moire.py` | 拍照/摩尔纹/几何形变模型 |
| `wechat_worst_case_compressor.py` | 微信最坏情况压缩 |
| `watermark_locator/v1/vv1_unet/train.py` | vv1 训练 |
| `watermark_locator/v1/vv2_yolov8/train.py` | vv2 训练 |
