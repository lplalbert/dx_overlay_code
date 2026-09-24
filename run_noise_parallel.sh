#!/bin/bash
# 多卡并行加噪: 复用 clean 树, 只跑 3 种噪声
# GPU 0 空闲 -> 6 个 worker;  GPU 2 负载最低 -> 2 个 worker
set -e
cd /data1/lpl/dx_overlay_code

PY=/data1/lpl/miniconda3/envs/lpl/bin/python
SCRIPT=watermark_locator/dataset/prepare_noise_from_clean.py
N=8

echo "launching $N shards ..."
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 $PY -u $SCRIPT --shard $i/$N \
    >> /data1/lpl/dx_overlay_code/noise_shard$i.log 2>&1 &
  echo "  shard $i/$N -> GPU 0  (pid $!)"
done
for i in 6 7; do
  CUDA_VISIBLE_DEVICES=2 $PY -u $SCRIPT --shard $i/$N \
    >> /data1/lpl/dx_overlay_code/noise_shard$i.log 2>&1 &
  echo "  shard $i/$N -> GPU 2  (pid $!)"
done

echo "waiting ..."
wait
echo "all shards done"
