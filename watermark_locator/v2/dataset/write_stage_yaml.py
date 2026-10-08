#!/usr/bin/env python3
"""为两阶段训练写出 YOLO 数据集配置。

训练方案（用户要求）::

    阶段一  clean + noisy  合起来训练
    阶段二  cropped        微调

``train_yolo.py`` 已经支持 ``cfg['stages']``，并把上一阶段的 ``best.pt`` 当下一
阶段的初始权重（``prev_best``），所以这里只要产出两份 data yaml 即可。

ultralytics 的 ``train:`` 接受 ``str`` 或 ``list``（``DATASET_KEY_TYPES`` 里写死
``"train": (str, list)``）。list 里的每一项按 ``path / x`` 解析 —— 给绝对路径时
pathlib 会直接取右侧，所以多棵树可以这样并列，**不需要 hardlink 合并树**::

    path: /data1/lpl/datasets_v2
    train:
      - /data1/lpl/datasets_v2/clean/images/train
      - /data1/lpl/datasets_v2/noisy/images/train

用法::

    python write_stage_yaml.py \\
        --clean  /data1/lpl/datasets_v2/clean \\
        --noisy  /data1/lpl/datasets_v2/noisy \\
        --cropped /data1/lpl/datasets_v2/cropped \\
        --out_dir /data1/lpl/datasets_v2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence


def _img_dir(root: str, split: str) -> str:
    d = os.path.join(root, 'images', split)
    return os.path.abspath(d)


def _count(d: str) -> int:
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))


def write_yaml(path: str, title: str, train_roots: Sequence[str],
               val_roots: Sequence[str]) -> dict:
    lines: List[str] = [
        f'# {title}',
        '# 自动生成：write_stage_yaml.py —— 改数据集请重跑，别手改',
        f'path: {os.path.abspath(os.path.dirname(path) or ".")}',
        'train:',
    ]
    for r in train_roots:
        lines.append(f'  - {r}')
    lines.append('val:')
    for r in val_roots:
        lines.append(f'  - {r}')
    lines += ['', 'names:', '  0: codeword', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))

    info = {
        'title': title,
        'yaml': os.path.abspath(path),
        'train': [{'root': r, 'n': _count(r)} for r in train_roots],
        'val': [{'root': r, 'n': _count(r)} for r in val_roots],
    }
    info['n_train'] = sum(x['n'] for x in info['train'])
    info['n_val'] = sum(x['n'] for x in info['val'])
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description='写两阶段训练的 YOLO data yaml')
    p.add_argument('--clean', required=True)
    p.add_argument('--noisy', required=True)
    p.add_argument('--cropped', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args(argv)

    if args.selftest:
        return selftest(args)

    os.makedirs(args.out_dir, exist_ok=True)

    s1 = write_yaml(
        os.path.join(args.out_dir, 'stage1.yaml'),
        '阶段一：clean + noisy（无噪声 + 微信压缩/拍照/拍照后微信压缩）',
        [_img_dir(args.clean, 'train'), _img_dir(args.noisy, 'train')],
        [_img_dir(args.clean, 'val'), _img_dir(args.noisy, 'val')])

    s2 = write_yaml(
        os.path.join(args.out_dir, 'stage2.yaml'),
        '阶段二：cropped 微调（不同尺寸裁剪，已 1:1 贴回 1920x1080）',
        [_img_dir(args.cropped, 'train')],
        [_img_dir(args.cropped, 'val')])

    for s in (s1, s2):
        print(f"\n{s['title']}")
        print(f"  {s['yaml']}")
        print(f"  train={s['n_train']}  val={s['n_val']}")
        for x in s['train']:
            print(f"    train {x['n']:>6}  {x['root']}")
        for x in s['val']:
            print(f"    val   {x['n']:>6}  {x['root']}")

    with open(os.path.join(args.out_dir, 'stages.json'), 'w') as f:
        json.dump({'stage1': s1, 'stage2': s2}, f, ensure_ascii=False, indent=2)
    return 0


def selftest(args) -> int:
    """ultralytics 必须真能吃下 list 形式的 train/val。"""
    print('=== write_stage_yaml selftest ===')
    ok = True
    try:
        from ultralytics.data.utils import check_det_dataset
    except Exception as e:  # pragma: no cover
        print(f'  [WARN] 无法导入 ultralytics ({e})，跳过解析校验')
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    s1 = write_yaml(
        os.path.join(args.out_dir, 'stage1.yaml'),
        'selftest stage1',
        [_img_dir(args.clean, 'train'), _img_dir(args.noisy, 'train')],
        [_img_dir(args.clean, 'val'), _img_dir(args.noisy, 'val')])
    s2 = write_yaml(
        os.path.join(args.out_dir, 'stage2.yaml'),
        'selftest stage2',
        [_img_dir(args.cropped, 'train')],
        [_img_dir(args.cropped, 'val')])

    for s in (s1, s2):
        try:
            d = check_det_dataset(s['yaml'])
        except Exception as e:
            print(f'  [FAIL] {s["yaml"]}: ultralytics 解析失败: {e}')
            ok = False
            continue
        for k in ('train', 'val'):
            v = d.get(k)
            lst = v if isinstance(v, list) else [v]
            missing = [x for x in lst if not os.path.isdir(x)]
            if missing:
                print(f'  [FAIL] {s["yaml"]} {k}: 目录不存在 {missing}')
                ok = False
            else:
                print(f'  [OK]   {os.path.basename(s["yaml"])} {k}: '
                      f'{len(lst)} 棵树 -> {sum(_count(x) for x in lst)} 张')
        # 硬门槛：list 必须被原样保留成 list，而不是被吃成单个路径
        if not isinstance(d.get('train'), list):
            print(f'  [FAIL] {s["yaml"]}: train 被解析成了 '
                  f'{type(d.get("train")).__name__}，多棵树没被保留')
            ok = False

    print('=== selftest:', 'PASS' if ok else 'FAIL', '===')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
