#!/usr/bin/env python3
"""核对 vv1/vv2 的 model.yaml 与预训练 .pt 的拓扑逐层一致。

防止 yaml 写歪却没发现（尤其 vv2 那份手抄的 YOLOv12）—— 只要有一行对不上，
`model: vv2/yolov12n.pt` 载入的就不是我们以为的那个网络。
"""
import os
import sys

import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(d):
    def cell(v):
        return str(v)
    return {
        'backbone': [[cell(x) for x in l] for l in d['backbone']],
        'head': [[cell(x) for x in l] for l in d['head']],
    }


def check(name, yaml_path, pt_path):
    mine = yaml.safe_load(open(yaml_path, encoding='utf-8'))
    print(f'— {name} —')
    print(f'  yaml: {os.path.relpath(yaml_path, HERE)}')
    if not os.path.exists(pt_path):
        print(f'  pt  : (缺 {os.path.relpath(pt_path, HERE)})  只验 yaml 能否构建')
        m = YOLO(yaml_path)
        n = sum(x.numel() for x in m.model.parameters())
        print(f'        params={n/1e6:.4f}M  nc={m.model.yaml.get("nc")}')
        return True

    m = YOLO(pt_path)
    got = m.model.yaml
    g, n = norm(got), norm(mine)
    ok = True
    for key in ('backbone', 'head'):
        same = g[key] == n[key]
        ok &= same
        print(f'  {key}: {"逐行一致" if same else "不一致"} '
              f'({len(g[key])} vs {len(n[key])} 层)')
        if not same:
            for i, (a, b) in enumerate(zip(g[key] + [[]], n[key] + [[]])):
                if a != b:
                    print(f'     [{i}] pt={a}')
                    print(f'         yaml={b}')
    n_par = sum(x.numel() for x in m.model.parameters())
    print(f'  params={n_par/1e6:.4f}M   pt.nc={got.get("nc")}  yaml.nc={mine.get("nc")}')
    if int(got.get('nc', -1)) != int(mine.get('nc', -1)):
        print('  注: nc 不同是预期的 (pt 是 COCO 80 类, yaml 是我们的 1 类);'
              ' 拓扑一致即可，ultralytics 会自动换掉 Detect 头')
    print(f'  {"OK" if ok else "FAIL"}')
    return ok


def main():
    ok = True
    ok &= check('vv1', os.path.join(HERE, 'vv1', 'model.yaml'),
                os.path.join(HERE, 'vv1', 'yolov8n.pt'))
    ok &= check('vv2', os.path.join(HERE, 'vv2', 'model.yaml'),
                os.path.join(HERE, 'vv2', 'yolov12n.pt'))
    print('\nPASS' if ok else '\nFAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
