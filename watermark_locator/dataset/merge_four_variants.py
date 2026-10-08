#!/usr/bin/env python
"""把 clean + 3 噪声四套裁剪图合并成单根, 便于一次训练吃下四个变体。

`build_multi_dataset(root)` / `build_yolo_dataset(data_root)` 都按
`root/{ds}/{train,val}/{images,masks,labels}` 扫描。四套变体的 stem 不冲突
(clean 无后缀, 噪声带 `_wechat` / `_pimog` / `_pimog_wechat`), 所以可以平铺进
同一个 `{ds}` 目录:

    datasets_labeled_3noise_cropped/
        clean/{ds}/...     4229 张
        noisy/{ds}/...    12687 张
        all/{ds}/...      16916 张   <- 本脚本产出, **硬链接**, 不占额外磁盘

之后 `data_root = .../cropped_all` 即可一次训练吃到全部四个变体。

用法:
    python watermark_locator/dataset/merge_four_variants.py
"""
import os

BASE = '/data1/lpl/datasets_labeled_3noise_cropped'
SRC_TREES = ('clean', 'noisy')
DST_TREE = 'all'
SUBS = ('images', 'masks', 'labels')


def main():
    n_link = n_skip = 0
    for ds in sorted(os.listdir(os.path.join(BASE, 'noisy'))):
        if not os.path.isdir(os.path.join(BASE, 'noisy', ds)):
            continue
        for split in ('train', 'val'):
            for sub in SUBS:
                dstd = os.path.join(BASE, DST_TREE, ds, split, sub)
                os.makedirs(dstd, exist_ok=True)
                for tree in SRC_TREES:
                    srcd = os.path.join(BASE, tree, ds, split, sub)
                    if not os.path.isdir(srcd):
                        continue
                    for fn in sorted(os.listdir(srcd)):
                        src = os.path.join(srcd, fn)
                        dst = os.path.join(dstd, fn)
                        if os.path.lexists(dst):
                            n_skip += 1
                            continue
                        os.link(src, dst)          # 硬链接, 同一文件系统
                        n_link += 1
        print(f'  {ds} done  (linked={n_link} skipped={n_skip})')

    print(f'\nhardlinked {n_link} files, skipped {n_skip} existing')
    print(f'merged root: {os.path.join(BASE, DST_TREE)}')

    for split in ('train', 'val'):
        n = {}
        for tree in SRC_TREES + (DST_TREE,):
            c = 0
            for ds in sorted(os.listdir(os.path.join(BASE, tree))):
                d = os.path.join(BASE, tree, ds, split, 'images')
                if os.path.isdir(d):
                    c += sum(1 for f in os.listdir(d) if f.lower().endswith('.png'))
            n[tree] = c
        print(f'  [{split}] clean={n["clean"]}  noisy={n["noisy"]}  all={n["all"]}'
              f'  (期望 all = clean + noisy)')


if __name__ == '__main__':
    main()
