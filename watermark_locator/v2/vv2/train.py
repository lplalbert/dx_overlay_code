#!/usr/bin/env python3
"""v2-vv2 训练入口：**唯一改了网络层**（YOLOv12 区域注意力 A2C2f）。

训练逻辑与 vv1 完全共用 ``../train_yolo.py``。改了哪些层、为什么有用、
论文依据，全部写在 ``model.yaml`` 的注释里。与 vv1 的唯一差别在
config 的 ``model`` 字段，其余超参逐字相同。

用法::

    python train.py                      # 用本目录 config.yaml
    python train.py --device 0 --batch 8
    python train.py --smoke              # 2 epoch 冒烟
    python train.py --resume output/.../weights/last.pt

先跑 ``../check_models.py`` 确认 model.yaml 与 yolov12n.pt 拓扑一致，
否则吃到的预训练权重不是你以为的那个网络。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import train_yolo  # noqa: E402

if __name__ == '__main__':
    train_yolo.main(['--config', os.path.join(HERE, 'config.yaml')] + sys.argv[1:])
