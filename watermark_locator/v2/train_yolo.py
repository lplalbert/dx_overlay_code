#!/usr/bin/env python3
"""v2 检测网络训练（vv1 / vv2 共用）。

与 v1 的 ``v1/vv2_yolov8/train.py`` 的差别，只有一条是本质的：

  **数据已经是最终训练窗**。``dataset/generate_dataset_v2.py`` 输出的每张图
  就是 1920×1080 的窗口，尺度覆盖 s∈[0.5,2.2] 靠**原生分辨率渲染后 1:1 取景**
  做进图里，不是靠 loader 缩放。所以这里不走 v1 的 multi_dataset 合并/缩放路径，
  直接吃现成的 YOLO 数据集目录 —— 也就彻底避开了 SCALE_MODE 那个把定位块拉大
  2× 的陷阱（见 DESIGN.md §12.4）。

  ``imgsz: 1920`` 是有意为之：ultralytics letterbox 到 1920×1920 时
  r = min(1920/1920, 1920/1080) = 1.0，**只 pad 不重采样**。
  若把 imgsz 降到 640，r=1/3，条纹周期 4px 会被下采样成 1.3px 而混叠 ——
  那是模板的渲染像素周期，不是屏幕分辨率周期。

**有效 batch 恒为 64**
    ultralytics 的 ``nbs``（nominal batch size）默认就是 64，梯度按
    ``accumulate = max(round(nbs / batch), 1)`` 累积。所以改 ``batch_size``
    只是墙钟加速/减速，有效 batch 不变，**lr 绝不能跟着 batch 放大**。
    这里显式传 ``nbs=64`` 把这个契约钉死在配置里。

vv1 / vv2 的唯一差别在 config 的 ``model``：
    vv1:  ``model: yolov8n.pt``        —— 零网络层改动，就是官方 yolov8n
    vv2:  ``model: vv2/model.yaml``    —— 只在 YAML 里换模块（YOLOv12 的
          A2C2f 区域注意力挂在 P4/P5），库代码一行不改

用法::

    python train_yolo.py --config vv1/config.yaml
    python train_yolo.py --config vv2/config.yaml --device 0
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import yaml

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(),
              logging.FileHandler('train_v2_yolo.log')],
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _stamp(data_yaml, image_size, train_length=0, val_length=0):
    """数据源指纹：只有真的变了才重建/换目录。"""
    with open(data_yaml) as f:
        ds = yaml.safe_load(f)
    root = os.path.abspath(ds.get('path', os.path.dirname(os.path.abspath(data_yaml))))
    n = 0
    for split in ('train', 'val'):
        d = os.path.join(root, ds.get(split, f'images/{split}').lstrip('./'))
        if os.path.isdir(d):
            n += sum(len(f) for _, _, f in os.walk(d))
    return f'{data_yaml}|{root}|{image_size[0]}x{image_size[1]}|{n}img' \
           f'|tr{int(train_length or 0)}|va{int(val_length or 0)}'


def _resolve_model(name, here):
    """裸文件名先按 vv 目录解析，再去 v2 根找。

    从别的 cwd 跑时 ``YOLO('yolov8n.pt')`` 会**静默去 GitHub 下载**，
    慢盘/断网环境下白等几分钟后才报错 —— 提前把路径钉死。
    """
    if not name:
        return name
    if os.path.exists(name):
        return os.path.abspath(name)
    for base in (here, os.path.dirname(here)):
        cand = os.path.join(base, name)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return name


def main(argv=None):
    p = argparse.ArgumentParser(description='v2 YOLO 检测训练 (vv1/vv2 共用)')
    p.add_argument('--config', required=True, help='配置文件路径')
    p.add_argument('--device', default=None, help='GPU 编号 (覆盖 config)')
    p.add_argument('--resume', default=None,
                   help='断点续训: 指向 runs/.../weights/last.pt (覆盖 config.resume)')
    p.add_argument('--batch', type=int, default=None, help='batch_size (覆盖 config)')
    p.add_argument('--workers', type=int, default=None, help='DataLoader worker 数 (覆盖 config)')
    p.add_argument('--smoke', action='store_true', help='冒烟: 每阶段压到 2 epoch / 小批量')
    args = p.parse_args(argv)

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    here = os.path.dirname(os.path.abspath(args.config))
    device = args.device or cfg.get('device', '0')
    resume_path = args.resume or cfg.get('resume') or None
    if resume_path and not os.path.exists(resume_path):
        raise FileNotFoundError(f'resume checkpoint not found: {resume_path}')

    batch_size = args.batch if args.batch is not None else int(cfg.get('batch_size', 8))
    # 慢盘上 worker 太少会饿死 GPU；太多只是互相抢同一块盘的寻道。
    workers = args.workers if args.workers is not None else int(
        cfg.get('workers', min(16, max(4, (os.cpu_count() or 8) // 4))))

    output_dir = cfg.get('output_dir', f'output/v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)

    seed = int(cfg.get('seed', 42))
    set_seed(seed)

    image_size = (cfg.get('image_height', 1080), cfg.get('image_width', 1920))
    model_name = _resolve_model(cfg.get('model', 'yolov8n.pt'), here)
    finetune_cfg = cfg.get('finetune') or {}

    stages = cfg.get('stages')
    if not stages:
        stages = [{
            'name': 'single',
            'data': cfg.get('data'),
            'epochs': cfg.get('epochs', 100),
            'lr': cfg.get('lr', 0.01),
        }]

    logger.info('=' * 60)
    logger.info(f'v2 YOLO  model={model_name}  imgsz={cfg.get("imgsz", 1920)}')
    logger.info(f'         batch={batch_size}  nbs={cfg.get("nbs", 64)}  '
                f'workers={workers}  device={device}')
    logger.info('=' * 60)

    # nbs 的语义要显式：有效 batch = nbs，不是 batch_size。
    nbs = int(cfg.get('nbs', 64))
    accum = max(round(nbs / batch_size), 1)
    logger.info(f'  梯度累积 {accum} 步 → 有效 batch {batch_size * accum}'
                f' (nbs={nbs})。**不要**按 batch_size 缩放 lr。')

    stage_summary = []
    prev_best = None

    for si, stage in enumerate(stages):
        tag = stage.get('tag', stage['name'])
        epochs = int(stage.get('epochs', cfg.get('epochs', 100)))
        lr = float(stage.get('lr', cfg.get('lr', 0.01)))
        patience = int(stage.get('patience', cfg.get('patience', 30)))
        data_yaml = stage.get('data') or cfg.get('data')
        if not data_yaml or not os.path.exists(data_yaml):
            raise FileNotFoundError(f'[{tag}] dataset yaml not found: {data_yaml}')
        if args.smoke:
            epochs = 2

        logger.info('')
        logger.info('=' * 60)
        logger.info(f'STAGE [{stage["name"]}]  data={data_yaml}')
        logger.info(f'         epochs={epochs}  lr={lr}  patience={patience}')
        logger.info('=' * 60)

        init = resume_path or prev_best or finetune_cfg.get('weight_path') or model_name
        if not (init and os.path.exists(init)):
            init = model_name
        logger.info(f'Loading model: {init}')
        if resume_path:
            logger.info(f'RESUME from {resume_path}')
        model = YOLO(init)

        run_name = f'yolo_{tag}'
        # optimizer=auto 会自行挑 lr 并**忽略 lr0** —— 配置里的 lr 就白写了。
        # 显式 SGD 才吃 stages 里的 lr（YOLO 惯例 lr0=0.01）。
        optimizer = stage.get('optimizer', cfg.get('optimizer', 'SGD'))
        kw = dict(
            data=data_yaml,
            epochs=epochs,
            imgsz=cfg.get('imgsz', 1920),
            # rect: 1920×1080 只 pad 到 stride 倍数 → 1920×1088。不透传的话
            # ultralytics 默认 rect=False，letterbox 成 1920×1920：43% 是 114
            # 灰边，白烧 1.77× 算力，还和 val 的 1920×1120 画布不一致。
            # r 恒为 1.0，两种都**不重采样**，160s×135s 的码字节距不受影响。
            # 我们的图全是 1920×1080 → batch_shapes 逐批全等，而 ultralytics
            # 只在 batch_shapes **不一致**时才因 rect 关掉 shuffle，所以 shuffle
            # 仍在（detect/train.py:get_dataloader）。
            rect=bool(cfg.get('rect', True)),
            batch=batch_size,
            nbs=nbs,
            workers=workers,
            lr0=lr,
            optimizer=optimizer,
            # weight_decay 同理必须透传：config 里写了但不进 kw 就等于没写。
            # 现值 5e-4 恰好等于 ultralytics 默认，改了才露馅。
            weight_decay=float(cfg.get('weight_decay', 0.0005)),
            device=device,
            project=output_dir,
            name=run_name,
            exist_ok=True,
            patience=patience,
            save=True,
            save_period=stage.get('save_every', cfg.get('save_every', 10)),
            seed=seed,
            verbose=True,
        )
        # channels_last: NHWC 布局，Ampere/Ada 上卷积免费提速，显存几乎不变。
        if cfg.get('channels_last'):
            kw['channels_last'] = bool(cfg['channels_last'])
        if resume_path:
            assert si == 0, 'resume 仅支持单阶段/首阶段'
            kw['resume'] = resume_path

        # 增广透传：config 的 `augment: {...}` 逐项覆盖 ultralytics 默认。
        # v2 的尺度已经在渲染时做进图里，这里通常要压掉几何/色彩增广：
        #   hsv: 0      色度/亮度抖动会直接糊掉 ±4 级的水印调制
        #   scale: 0    尺度增广会破坏"格点间距 = 160s/135s"的先验
        #   fliplr: 0   45° 条纹翻转后变成 135°，不再是训练里的模式
        aug = {**(cfg.get('augment') or {}), **(stage.get('augment') or {})}
        kw.update(aug)
        if aug:
            logger.info(f'  augment overrides: {aug}')
        model.train(**kw)

        # ultralytics 自己拼 save_dir，找不到 best.pt 阶段间权重链就断了
        save_dir = str(getattr(model.trainer, 'save_dir', None)
                       or os.path.join(output_dir, run_name))
        best_pt = os.path.join(save_dir, 'weights', 'best.pt')
        last_pt = os.path.join(save_dir, 'weights', 'last.pt')
        if os.path.exists(best_pt):
            prev_best = best_pt
        elif os.path.exists(last_pt):
            prev_best = last_pt

        stage_summary.append({
            'name': stage['name'], 'data': data_yaml,
            'epochs': epochs, 'lr': lr, 'batch': batch_size, 'nbs': nbs,
            'best_weights': best_pt if os.path.exists(best_pt) else None,
            'last_weights': last_pt if os.path.exists(last_pt) else None,
            'save_dir': save_dir,
        })

    summary = {
        'stages': stage_summary,
        'best_weights': prev_best,
        'total_epochs': sum(s['epochs'] for s in stage_summary),
        'config': cfg,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info('=' * 60)
    logger.info('Training complete!')
    for s in stage_summary:
        logger.info(f'  [{s["name"]}] {s["epochs"]} ep  best={s["best_weights"]}')
    logger.info(f'Best model: {prev_best}')
    logger.info(f'Output: {output_dir}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
