#!/usr/bin/env bash
# v2 三个数据集一次生成（clean → noisy → cropped），可断点续跑。
#
#   1 clean    无噪声                         N=NUM_SAMPLES
#   2 noisy    微信压缩 / 拍照模拟 /          3×N
#              拍照后微信压缩
#   3 cropped  (clean ∪ noisy) 1:1 裁剪       4×N
#              后重框 1920×1080
#
# 三步都"跳过已存在"，中断后重跑即续。分片只改墙钟不改结果
# （crop 侧是逐样本种子，见 prepare_crop_aug_v2.collect_sources）。
#
# noisy 走 --shard 分片；cropped 的生成器改用 --jobs 内部并行（**没有**
# --shard），所以第三步是一次调用，不是分片循环。
#
# 用法:
#   bash dataset/run_three_datasets.sh
#   NUM_SAMPLES=4000 CLEAN_JOBS=8 NOISE_SHARDS=4 CROP_JOBS=6 \
#       bash dataset/run_three_datasets.sh
set -euo pipefail

PY=${PY:-/data1/lpl/miniconda3/envs/lpl/bin/python}
ROOT=${ROOT:-/data1/lpl/datasets_v2}
NUM_SAMPLES=${NUM_SAMPLES:-4000}
CLEAN_JOBS=${CLEAN_JOBS:-8}
NOISE_SHARDS=${NOISE_SHARDS:-4}
CROP_JOBS=${CROP_JOBS:-6}
GPUS=${GPUS:-"0 1 2 3"}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"

log() { printf '\n\033[1m════ %s ════\033[0m\n' "$*"; }

# ── 1. clean ─────────────────────────────────────────────────────────
log "1/3  clean   N=$NUM_SAMPLES  jobs=$CLEAN_JOBS  → $ROOT/clean"
if [ -f "$ROOT/clean/manifest.json" ]; then
  echo "已存在 $ROOT/clean/manifest.json，跳过（要重跑请删掉它）"
else
  $PY generate_dataset_v2.py \
      --num_samples "$NUM_SAMPLES" --no_noise \
      --carrier_root /data1/lpl/datasets \
      --output_dir "$ROOT/clean" --jobs "$CLEAN_JOBS"
fi

# ── 2. noisy（3 变体，分片）─────────────────────────────────────────
log "2/3  noisy   3 × clean   shards=$NOISE_SHARDS  → $ROOT/noisy"
pids=()
# 轮转：第 s 个分片落在第 (s % |GPUS|) 张卡上
set -- $GPUS
NG=$#
s=0
while [ $s -lt "$NOISE_SHARDS" ]; do
  gidx=$((s % NG))
  g=$(eval "printf '%s' \${$((gidx + 1))}")
  CUDA_VISIBLE_DEVICES=$g $PY prepare_noise_v2.py \
      --clean_root "$ROOT/clean" --out_root "$ROOT/noisy" \
      --shard "$s/$NOISE_SHARDS" &
  pids+=($!)
  s=$((s + 1))
done
for p in "${pids[@]}"; do wait "$p"; done

# ── 3. cropped（1:1 裁 + 重框）────────────────────────────────────
# cropped 由 clean ∪ noisy 两棵树裁出，所以 --src_root 要给两次。
# 生成器用 --jobs 内部并行，**没有** --shard。
log "3/3  cropped 1:1 裁剪 + 重框   jobs=$CROP_JOBS  → $ROOT/cropped"
$PY prepare_crop_aug_v2.py \
    --src_root "$ROOT/clean" \
    --src_root "$ROOT/noisy" \
    --output_dir "$ROOT/cropped" \
    --carrier_root /data1/lpl/datasets \
    --jobs "$CROP_JOBS"

log "完成"
ls -d "$ROOT"/{clean,noisy,cropped}
for t in clean noisy cropped; do
  n=$(find "$ROOT/$t/images" -name '*.png' 2>/dev/null | wc -l)
  printf '  %-8s %6d img\n' "$t" "$n"
done
echo
echo "stage1: $ROOT/stage1.yaml   (clean + noisy → 打底)"
echo "stage2: $ROOT/stage2.yaml   (cropped → 微调)"
