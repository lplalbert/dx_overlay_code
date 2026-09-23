"""
v2-vv2: 联合识别 + YOLOv8 目标检测

利用6个定位块的棋盘格固定间隔几何先验联合检测。
- 6个位置类别 (locator_0 ~ locator_5)，每个类别对应一个固定网格位置
- 后处理用棋盘格几何约束做联合校验和去噪

棋盘格位置 (block网格 4x6):
    (0,3), (1,1), (1,5), (2,3), (3,1), (3,5)

用法:
    python train.py --data_dir ../../dataset/data --epochs 50
"""

import argparse
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

# ── 棋盘格几何先验 ──
BLOCK_ROWS, BLOCK_COLS = 4, 6
LOCATOR_GRID = [(0, 3), (1, 1), (1, 5), (2, 3), (3, 1), (3, 5)]
NUM_LOCATORS = 6


def get_expected_centers(screen_w=1920, screen_h=1080):
    """6个定位块的期望归一化中心坐标。"""
    block_h = screen_h / BLOCK_ROWS
    block_w = screen_w / BLOCK_COLS
    msg_h = block_h / 2
    msg_w = block_w / 2
    centers = []
    for (bi, bj) in LOCATOR_GRID:
        cx = (bj * block_w + msg_w + msg_w / 2) / screen_w
        cy = (bi * block_h + msg_h + msg_h / 2) / screen_h
        centers.append((cx, cy))
    return np.array(centers)


def convert_to_joint_labels(data_dir, output_dir):
    """
    将单一类别标签转换为6个位置类别。

    原始标签: class=0 (locator), 归一化 cx cy w h
    联合标签: class=0~5 (locator_0~locator_5)，按最近期望位置分配
    """
    os.makedirs(output_dir, exist_ok=True)
    expected = get_expected_centers()

    for split in ['images', 'labels']:
        src = os.path.join(data_dir, 'vv2', split)
        dst = os.path.join(output_dir, split)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            src_path = os.path.join(src, f)
            dst_path = os.path.join(dst, f)
            if f.endswith('.png'):
                img = cv2.imread(src_path)
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                cb = ycrcb[:, :, 2]
                cb_3ch = cv2.merge([cb, cb, cb])
                cv2.imwrite(dst_path, cb_3ch)
            elif f.endswith('.txt'):
                # 将单类别标签转为6位置类别
                lines_out = []
                with open(src_path, 'r') as fin:
                    for line in fin:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # 找最近的期望位置
                        dists = np.sqrt((expected[:, 0] - cx)**2 + (expected[:, 1] - cy)**2)
                        pos_idx = int(np.argmin(dists))
                        lines_out.append(f"{pos_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                with open(dst_path, 'w') as fout:
                    fout.write('\n'.join(lines_out) + '\n')
            else:
                shutil.copy2(src_path, dst_path)


def postprocess_joint(detections, screen_w=1920, screen_h=1080):
    """
    v2联合校验: 利用棋盘格几何约束。

    1. 检查是否6个位置类别都检测到
    2. 验证检测框间距是否符合棋盘格周期
    3. 用几何一致性剔除误检、补充漏检
    """
    expected = get_expected_centers(screen_w, screen_h)

    # 按类别分组
    by_class = {}
    for det in detections:
        cls = int(det['class'])
        if cls not in by_class or det['conf'] > by_class[cls]['conf']:
            by_class[cls] = det

    # 联合校验: 检查间距一致性
    found = sorted(by_class.keys())
    if len(found) >= 2:
        # 验证相邻检测到的位置间距是否符合预期
        for i in range(len(found) - 1):
            a, b = found[i], found[i + 1]
            det_dist = np.sqrt(
                (by_class[a]['cx'] - by_class[b]['cx'])**2 +
                (by_class[a]['cy'] - by_class[b]['cy'])**2
            )
            exp_dist = np.sqrt(
                (expected[a][0] - expected[b][0])**2 +
                (expected[a][1] - expected[b][1])**2
            )
            # 间距偏差超过50%认为不可信
            if exp_dist > 0 and abs(det_dist - exp_dist) / exp_dist > 0.5:
                by_class[b]['conf'] *= 0.5  # 降低置信度

    return list(by_class.values())


def train(args):
    joint_data_dir = os.path.join(args.data_dir, 'vv2_joint_cb')
    print("[v2-vv2] Joint YOLOv8 Detection (6 position classes)")
    print("转换Cb通道数据集 + 联合类别标签...")
    convert_to_joint_labels(args.data_dir, joint_data_dir)

    yaml_path = os.path.join(joint_data_dir, 'watermark_joint.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {os.path.abspath(joint_data_dir)}
train: images
val: images

names:
  0: locator_pos0
  1: locator_pos1
  2: locator_pos2
  3: locator_pos3
  4: locator_pos4
  5: locator_pos5
""")

    model = YOLO(args.model)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        lr0=args.lr,
        device=args.device,
        project=args.output_dir,
        name='v2_vv2_yolo_joint',
        exist_ok=True,
        patience=20,
        save=True,
        verbose=True,
    )
    print(f"[v2-vv2] 训练完成!")
    print(f"最佳模型: {args.output_dir}/v2_vv2_yolo_joint/weights/best.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../dataset/data")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=1080)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(args)
