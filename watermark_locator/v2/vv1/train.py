#!/usr/bin/env python3
"""v2-vv1 训练入口：**零网络层改动**（官方 yolov8n）。

训练逻辑与 vv2 完全共用 ``../train_yolo.py``。vv1/vv2 的唯一差别在
config 的 ``model`` 字段：

    vv1  model: vv1/yolov8n.pt    ← 拓扑 = ultralytics 官方 yolov8.yaml
    vv2  model: vv2/yolov12n.pt   ← 拓扑 = YOLOv12 (A2C2f 区域注意力)

数据、imgsz、nbs、lr、seed、增广设置全部逐字相同，所以两边 mAP 的差异
只能来自网络结构。

用法::

    python train.py                      # 用本目录 config.yaml
    python train.py --device 0 --batch 8
    python train.py --smoke              # 2 epoch 冒烟
    python train.py --resume output/.../weights/last.pt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import train_yolo  # noqa: E402

if __name__ == '__main__':
    train_yolo.main(['--config', os.path.join(HERE, 'config.yaml')] + sys.argv[1:])
