"""
水印定位检测网络

参考 YOLO 架构设计：
- 轻量 Backbone 提取特征
- 多尺度检测头（单块 / 2×2块组 / 全局区域）
- 特征融合模块（多块加权聚合）
- 解码头（从融合特征解码码字）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────── Backbone ─────────────────────

class ConvBnSiLU(nn.Module):
    """Conv + BatchNorm + SiLU 激活（YOLO标准组件）。"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3k2Block(nn.Module):
    """
    C3k2 模块（简化版 CSP Bottleneck）。
    参考 YOLO11 的 C3k2 设计：split → 两个分支处理 → concat。
    """
    def __init__(self, in_ch, out_ch, num_bottlenecks=2, k=3):
        super().__init__()
        mid_ch = out_ch // 2
        self.cv1 = ConvBnSiLU(in_ch, mid_ch, 1, 1, 0)
        self.cv2 = ConvBnSiLU(in_ch, mid_ch, 1, 1, 0)
        self.bottlenecks = nn.Sequential(*[
            ConvBnSiLU(mid_ch, mid_ch, k, 1, k // 2)
            for _ in range(num_bottlenecks)
        ])
        self.cv3 = ConvBnSiLU(mid_ch * 2, out_ch, 1, 1, 0)

    def forward(self, x):
        y1 = self.bottlenecks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class WatermarkBackbone(nn.Module):
    """
    轻量 Backbone：4层卷积逐步下采样。

    输入: (B, 3, H, W)  RGB 3通道图像（载体+水印+噪声的混合体）
    输出: (B, 128, H/16, W/16)  特征图

    对于 1920×1080 输入：
      → (B, 128, 68, 120)

    设计说明：
    - 输入是拍照+微信压缩后的RGB图像（载体内容+水印叠加）
    - 第一层用 stride=2 大幅降低分辨率，后续用 C3k2 提取语义特征
    - 水印信号非常微弱（alpha≈0.05），需要网络学会从载体内容中分离水印
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.layer1 = ConvBnSiLU(in_ch, 16, kernel_size=3, stride=2, padding=1)   # /2
        self.layer2 = C3k2Block(16, 32, num_bottlenecks=2)                         # /2
        self.layer3 = C3k2Block(32, 64, num_bottlenecks=2)                         # /2
        self.layer4 = C3k2Block(64, 128, num_bottlenecks=2)                        # /2

    def forward(self, x):
        x = self.layer1(x)   # (B,16, H/2, W/2)
        x = F.max_pool2d(x, 2)  # 额外下采样
        x = self.layer2(x)   # (B,32, H/4, W/4)
        x = F.max_pool2d(x, 2)
        x = self.layer3(x)   # (B,64, H/8, W/8)
        x = F.max_pool2d(x, 2)
        x = self.layer4(x)   # (B,128, H/16, W/16)
        return x


# ───────────────────── 检测头 ─────────────────────

class ScaleDetectHead(nn.Module):
    """
    单尺度检测头（类 YOLO 解耦头）。

    通过 AdaptiveAvgPool2d 将特征图对齐到目标网格尺寸，
    然后预测每个网格位置的 [conf, dx, dy, scale]。

    注意：当 grid_h=grid_w=1 时（全局尺度），使用 GroupNorm 替代 BatchNorm
    以避免 batch_size=1 时 BN 的单值问题。
    """
    def __init__(self, in_ch: int, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))

        # 当空间尺寸为1×1时，用GroupNorm替代BatchNorm
        if grid_h == 1 and grid_w == 1:
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, 64, 1, bias=False),
                nn.GroupNorm(1, 64),  # LayerNorm等价
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 4, 1),
            )
        else:
            self.head = nn.Sequential(
                ConvBnSiLU(in_ch, 64, 1, 1, 0),
                nn.Conv2d(64, 4, 1),  # [conf, dx, dy, scale]
            )

    def forward(self, feat):
        """
        Args:
            feat: (B, C, H, W) backbone特征图

        Returns:
            (B, 4, grid_h, grid_w) 检测输出
            [0] conf: 定位块置信度 (sigmoid)
            [1] dx: x方向偏移 (tanh, 相对块宽)
            [2] dy: y方向偏移 (tanh, 相对块高)
            [3] scale: 尺度修正 (sigmoid*0.5+0.75, 约0.75~1.25)
        """
        x = self.pool(feat)
        out = self.head(x)
        # 分别激活
        out_conf = torch.sigmoid(out[:, 0:1])           # [0, 1]
        out_offset = torch.tanh(out[:, 1:3])            # [-1, 1]
        out_scale = torch.sigmoid(out[:, 3:4]) * 0.5 + 0.75  # [0.75, 1.25]
        return torch.cat([out_conf, out_offset, out_scale], dim=1)


class MultiScaleDetector(nn.Module):
    """
    多尺度检测器（参考 YOLO P3/P4/P5 设计）。

    三个检测尺度：
    - Scale 1 (4×6): 单块定位，最精细
    - Scale 2 (2×3): 2×2块组，中等精度
    - Scale 3 (1×1): 全局区域，确认水印存在
    """
    def __init__(self, in_ch=128, block_rows=4, block_cols=6):
        super().__init__()
        self.head_s1 = ScaleDetectHead(in_ch, block_rows, block_cols)         # 4×6
        self.head_s2 = ScaleDetectHead(in_ch, block_rows // 2, block_cols // 2)  # 2×3
        self.head_s3 = ScaleDetectHead(in_ch, 1, 1)                           # 1×1

    def forward(self, feat):
        """
        Returns:
            dict with keys 's1', 's2', 's3'
            每个值: (B, 4, grid_h, grid_w)
        """
        return {
            's1': self.head_s1(feat),
            's2': self.head_s2(feat),
            's3': self.head_s3(feat),
        }


# ───────────────────── 特征融合 ─────────────────────

class BlockFeatureFusion(nn.Module):
    """
    多块特征融合模块。

    从 backbone 特征图中，根据检测到的定位块位置，
    提取 ROI 特征并加权聚合。

    类比：YOLO 的 FPN 融合不同尺度特征，
    这里融合同一码字在不同空间位置的特征（利用重复性）。
    """
    def __init__(self, in_ch=128, out_ch=128):
        super().__init__()
        self.reduce = ConvBnSiLU(in_ch, out_ch, 1, 1, 0)
        self.fuse_conv = nn.Sequential(
            ConvBnSiLU(out_ch, out_ch, 3, 1, 1),
            ConvBnSiLU(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, feat, detection_s1, block_rows=4, block_cols=6):
        """
        Args:
            feat: (B, C, H, W) backbone特征图
            detection_s1: (B, 4, block_rows, block_cols) Scale1检测结果

        Returns:
            (B, block_rows*block_cols, out_ch) 每个块位置的融合特征
        """
        B, C, H, W = feat.shape
        feat_reduced = self.reduce(feat)  # (B, out_ch, H, W)

        # 自适应池化到块网格
        feat_grid = F.adaptive_avg_pool2d(feat_reduced, (block_rows, block_cols))
        # feat_grid: (B, out_ch, block_rows, block_cols)

        # 用检测置信度加权
        conf = detection_s1[:, 0:1]  # (B, 1, block_rows, block_cols)
        feat_weighted = feat_grid * conf  # 高置信度位置特征增强

        # 也从2×2块组特征中提取上下文
        feat_coarse = F.adaptive_avg_pool2d(feat_reduced, (block_rows // 2, block_cols // 2))
        feat_coarse_up = F.interpolate(feat_coarse, size=(block_rows, block_cols), mode='bilinear', align_corners=True)

        # 融合精细和粗糙特征
        feat_fused = feat_weighted + feat_coarse_up
        feat_fused = self.fuse_conv(feat_fused.unsqueeze(0) if feat_fused.dim() == 3 else feat_fused)

        # reshape: (B, out_ch, R, C) → (B, R*C, out_ch)
        feat_flat = feat_fused.flatten(2).permute(0, 2, 1)  # (B, R*C, out_ch)
        return feat_flat


# ───────────────────── 解码头 ─────────────────────

class CodewordDecoder(nn.Module):
    """
    码字解码头。

    从融合特征中解码每个块位置的码字（0~15）或定位图案（16）。
    """
    def __init__(self, in_ch=128, num_classes=17, num_blocks=24):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, num_blocks, in_ch)

        Returns:
            (B, num_blocks, num_classes) 码字logits
        """
        return self.decoder(fused_features)


# ───────────────────── 完整网络 ─────────────────────

class WatermarkLocatorNet(nn.Module):
    """
    水印定位+解码一体化网络。

    流程：
    1. Backbone 提取 Cb 通道特征
    2. 多尺度检测头 → 定位块位置
    3. 特征融合 → 多块特征加权聚合
    4. 解码头 → 每个块的码字概率

    输出：
    - detection: dict {'s1', 's2', 's3'}, 每个 (B, 4, grid_h, grid_w)
    - codeword_logits: (B, 24, 17)
    """
    def __init__(self, in_ch=3, block_rows=4, block_cols=6, num_classes=17):
        super().__init__()
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.num_blocks = block_rows * block_cols

        self.backbone = WatermarkBackbone(in_ch=in_ch)
        self.detector = MultiScaleDetector(
            in_ch=128, block_rows=block_rows, block_cols=block_cols
        )
        self.fusion = BlockFeatureFusion(in_ch=128, out_ch=128)
        self.decoder = CodewordDecoder(
            in_ch=128, num_classes=num_classes, num_blocks=self.num_blocks
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB 3通道图像（载体+水印+噪声混合）

        Returns:
            dict:
              'detections': {'s1': (B,4,4,6), 's2': (B,4,2,3), 's3': (B,4,1,1)}
              'codeword_logits': (B, 24, 17)
        """
        feat = self.backbone(x)
        detections = self.detector(feat)
        fused = self.fusion(feat, detections['s1'],
                            self.block_rows, self.block_cols)
        codeword_logits = self.decoder(fused)
        return {
            'detections': detections,
            'codeword_logits': codeword_logits,
        }


# ───────────────────── 损失函数 ─────────────────────

class FocalLoss(nn.Module):
    """Focal Loss 处理正负样本不平衡。"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: (N,) 预测概率 [0,1]
            target: (N,) 真实标签 {0, 1}
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class WatermarkLocatorLoss(nn.Module):
    """
    多任务损失函数。

    L = λ1 * L_focal(det) + λ2 * L1(offset/scale) + λ3 * CE(codeword)
    """
    def __init__(self, lambda_det=1.0, lambda_offset=0.5, lambda_decode=1.0):
        super().__init__()
        self.focal = FocalLoss()
        self.l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_det = lambda_det
        self.lambda_offset = lambda_offset
        self.lambda_decode = lambda_decode

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from WatermarkLocatorNet.forward()
            targets: dict {
                'locator_map': (B, 4, 6),    定位块标签 {0, 1}
                'offset_map': (B, 4, 6, 2),  亚像素偏移 [-0.5, 0.5]
                'scale_map': (B, 4, 6),       尺度标签 (≈1.0)
                'codeword_labels': (B, 24),   码字标签 {0..16}
            }
        """
        det_s1 = predictions['detections']['s1']  # (B, 4, 4, 6)

        # --- 检测损失 ---
        pred_conf = det_s1[:, 0].reshape(-1)          # (B*4*6,)
        gt_conf = targets['locator_map'].reshape(-1)   # (B*4*6,)
        loss_det = self.focal(pred_conf, gt_conf.float())

        # --- 偏移损失（只在正样本位置计算）---
        mask = targets['locator_map'] > 0.5  # (B, 4, 6)
        if mask.sum() > 0:
            pred_dx = det_s1[:, 1]  # (B, 4, 6)
            pred_dy = det_s1[:, 2]
            pred_scale = det_s1[:, 3]
            gt_dx = targets['offset_map'][..., 0]
            gt_dy = targets['offset_map'][..., 1]
            gt_scale = targets['scale_map']

            loss_offset = (
                self.l1(pred_dx[mask], gt_dx[mask]) +
                self.l1(pred_dy[mask], gt_dy[mask]) +
                self.l1(pred_scale[mask], gt_scale[mask])
            ) * self.lambda_offset
        else:
            loss_offset = torch.tensor(0.0, device=det_s1.device)

        # --- 解码损失 ---
        pred_logits = predictions['codeword_logits']  # (B, 24, 17)
        gt_codewords = targets['codeword_labels']      # (B, 24)
        B, N, C = pred_logits.shape
        loss_decode = self.ce(
            pred_logits.reshape(B * N, C),
            gt_codewords.reshape(B * N)
        )

        total = self.lambda_det * loss_det + loss_offset + self.lambda_decode * loss_decode
        return {
            'total': total,
            'det': loss_det,
            'offset': loss_offset,
            'decode': loss_decode,
        }


if __name__ == "__main__":
    # 测试网络结构
    model = WatermarkLocatorNet(in_ch=1, block_rows=4, block_cols=6)
    x = torch.randn(2, 1, 1080, 1920)
    out = model(x)
    print("Detection S1 shape:", out['detections']['s1'].shape)
    print("Detection S2 shape:", out['detections']['s2'].shape)
    print("Detection S3 shape:", out['detections']['s3'].shape)
    print("Codeword logits shape:", out['codeword_logits'].shape)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
