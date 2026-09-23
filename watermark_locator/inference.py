"""
水印定位器 — 推理/检测脚本

从截屏图像中定位水印并解码 watermark_id。

用法:
    python inference.py --input screenshot.png --model checkpoints/best_model.pth
    python inference.py --input_dir screenshots/ --model checkpoints/best_model.pth
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

from network import WatermarkLocatorNet
from utils import (
    rs_decode,
    checkerboard_locator_positions,
    locator_absolute_positions,
    nms_with_checkerboard_prior,
    grid_alignment_from_detections,
    draw_detections,
    LOCATOR_CODEWORD_INDEX,
)


def load_model(model_path: str, block_rows: int = 4, block_cols: int = 6,
               device: str = 'cpu') -> WatermarkLocatorNet:
    """加载训练好的模型。"""
    model = WatermarkLocatorNet(
        in_ch=3,
        block_rows=block_rows,
        block_cols=block_cols,
        num_classes=17,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def prepare_rgb_input(image_bgr: np.ndarray) -> np.ndarray:
    """BGR图像转RGB并归一化到[0,1]。"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return image_rgb


@torch.no_grad()
def detect_watermark(
    model: WatermarkLocatorNet,
    rgb_image: np.ndarray,
    device: str = 'cpu',
    conf_threshold: float = 0.5,
) -> dict:
    """
    检测单张图像中的水印。

    Returns:
        dict with:
            'detected': bool
            'detections': list of detected locator blocks
            'codeword_probs': (24, 17) codeword probabilities
            'watermark_id': decoded ID (or -1 if failed)
            'global_conf': Scale3 confidence
    """
    H, W = rgb_image.shape[:2]

    # 转为tensor: (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device, dtype=torch.float32)

    # 前向推理
    outputs = model(tensor)

    # --- Scale 3: 全局检测 ---
    s3 = outputs['detections']['s3'][0]  # (4, 1, 1)
    global_conf = s3[0, 0, 0].item()

    if global_conf < 0.3:
        return {
            'detected': False,
            'detections': [],
            'codeword_probs': None,
            'watermark_id': -1,
            'global_conf': global_conf,
        }

    # --- Scale 1: 单块定位 ---
    s1 = outputs['detections']['s1'][0]  # (4, 4, 6)
    block_rows, block_cols = s1.shape[1], s1.shape[2]
    block_w = W / block_cols
    block_h = H / block_rows

    detections = []
    for row in range(block_rows):
        for col in range(block_cols):
            conf = s1[0, row, col].item()
            dx = s1[1, row, col].item()
            dy = s1[2, row, col].item()
            scale = s1[3, row, col].item()

            if conf > conf_threshold:
                # 计算精确位置
                cx = (col + 0.5 + dx) * block_w
                cy = (row + 0.5 + dy) * block_h
                detections.append({
                    'cx': cx, 'cy': cy,
                    'conf': conf,
                    'dx': dx, 'dy': dy,
                    'scale': scale,
                    'grid_row': row, 'grid_col': col,
                })

    # --- 棋盘先验NMS ---
    detections = nms_with_checkerboard_prior(detections, block_w, block_h)

    # --- 网格对齐 ---
    M = grid_alignment_from_detections(
        detections, block_rows, block_cols, W, H
    )

    # --- 码字解码 ---
    codeword_probs = torch.softmax(outputs['codeword_logits'][0], dim=-1)  # (24, 17)
    codeword_probs_np = codeword_probs.cpu().numpy()

    # 解码
    wm_id, decode_ok = rs_decode(codeword_probs_np)

    return {
        'detected': len(detections) > 0,
        'detections': detections,
        'codeword_probs': codeword_probs_np,
        'watermark_id': wm_id,
        'decode_ok': decode_ok,
        'global_conf': global_conf,
        'alignment_matrix': M,
    }


def process_image(
    model: WatermarkLocatorNet,
    image_path: str,
    device: str = 'cpu',
    output_dir: str = None,
) -> dict:
    """处理单张图像。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load {image_path}")
        return None

    rgb = prepare_rgb_input(image)
    result = detect_watermark(model, rgb, device)

    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"Size: {image.shape[1]}×{image.shape[0]}")
    print(f"Global confidence: {result['global_conf']:.4f}")
    print(f"Detected: {result['detected']}")
    print(f"Locator blocks found: {len(result['detections'])}")

    if result['decode_ok']:
        print(f"Watermark ID: {result['watermark_id']}")
    else:
        print(f"Decode: FAILED")

    # 可视化
    if output_dir and result['detected']:
        H, W = rgb.shape[:2]
        block_w = W / 6
        block_h = H / 4
        vis_bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        vis = draw_detections(
            vis_bgr, result['detections'], block_w, block_h,
        )
        out_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, vis)
        print(f"Saved visualization: {out_path}")

    return result


def process_directory(
    model: WatermarkLocatorNet,
    input_dir: str,
    device: str = 'cpu',
    output_dir: str = None,
) -> list:
    """处理目录中的所有图像。"""
    results = []
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    images = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(extensions)
    ])

    print(f"Found {len(images)} images in {input_dir}")

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        result = process_image(model, img_path, device, output_dir)
        if result:
            result['image'] = img_name
            results.append(result)

    # 统计
    total = len(results)
    detected = sum(1 for r in results if r['detected'])
    decoded = sum(1 for r in results if r.get('decode_ok', False))
    print(f"\n{'='*60}")
    print(f"Summary: {detected}/{total} detected, {decoded}/{total} decoded")

    return results


def main():
    parser = argparse.ArgumentParser(description="Watermark Locator Inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Single input image path")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of input images")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="Output directory for visualizations")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--block_rows", type=int, default=4)
    parser.add_argument("--block_cols", type=int, default=6)
    parser.add_argument("--save_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Either --input or --input_dir must be provided")

    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # 加载模型
    model = load_model(args.model, args.block_rows, args.block_cols, device)
    print(f"Loaded model from {args.model}")

    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # 推理
    if args.input:
        result = process_image(model, args.input, device, args.output_dir)
        results = [result] if result else []
    else:
        results = process_directory(model, args.input_dir, device, args.output_dir)

    # 保存JSON
    if args.save_json and results:
        # 只保存可序列化的字段
        json_results = []
        for r in results:
            json_results.append({
                'image': r.get('image', ''),
                'detected': r['detected'],
                'global_conf': r['global_conf'],
                'watermark_id': r['watermark_id'],
                'decode_ok': r.get('decode_ok', False),
                'num_detections': len(r['detections']),
            })
        with open(args.save_json, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nSaved results to {args.save_json}")


if __name__ == "__main__":
    main()
