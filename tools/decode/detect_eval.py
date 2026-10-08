"""检测与框评估的公共口径 —— 从 imgsz_test.py 抽出的可复用部分。

为什么要单独抽
--------------
选权重时要在多个 checkpoint 上用**同一套** detect/eval 口径比较, 否则数字不可比。
原实现散在 imgsz_test.py 里, 而那个文件为了 GT 常量一路 import 到 rect_and_detect.py,
把整堆一次性探针都拖了进来。这里只留纯函数, 不碰任何数据路径。

**detect_at 的缩放/pad 口径必须严格一致**, 改这里就等于改了所有历史对比:
    1. 按 max(w,h) 缩到 imgsz, INTER_AREA
    2. 右/下 pad 到 32 的倍数, 填充 114 (YOLO 默认灰)
    3. model.predict(imgsz=max(pw,ph))
    4. 预测框除以 s 映回原图坐标
"""
import cv2
import numpy as np


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(boxes, scores, thr=0.5):
    """跨尺度合并用: 按分数降序贪心 NMS。"""
    o = np.argsort(scores)[::-1]
    keep = []
    for i in o:
        if all(iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return keep


def detect_at(model, img, imgsz, conf=0.25):
    """按 imgsz 跑一次, 框映回原图坐标。imgsz 越大, 输入里定位块越大。"""
    h, w = img.shape[:2]
    s = imgsz / float(max(w, h))
    nw, nh = int(round(w * s)), int(round(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pw, ph = (nw + 31) // 32 * 32, (nh + 31) // 32 * 32
    canvas = np.full((ph, pw, 3), 114, np.uint8)
    canvas[:nh, :nw] = small
    r = model.predict(source=canvas, imgsz=max(pw, ph), conf=conf, verbose=False)[0]
    boxes, scores = [], []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append((x1 / s, y1 / s, x2 / s, y2 / s))
            scores.append(float(b.conf[0]))
    return boxes, scores


def eval_boxes(pred, gt):
    """贪心一对一匹配 (IoU>=0.5)。返回 (TP, FP, FN)。"""
    used, tp = set(), 0
    for g in gt:
        best, bi = -1.0, -1
        for i, p in enumerate(pred):
            if i in used:
                continue
            v = iou(g, p)
            if v > best:
                best, bi = v, i
        if best >= 0.5 and bi >= 0:
            used.add(bi)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp
