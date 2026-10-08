# tools/decode — 解码率这把尺子

匹配滤波解码器 + 一组用它做的判据实验。**检测器不在环**, 纯 CPU。

## 为什么要有它

水印能不能读出来, 得按**解码率**定, 不能按残余(零偏移相关)定 —— 残余高的档解码率
可以是 0, 这已经证伪过。所以档位上界、噪声仿真参数、实拍链的上界, 都要拿解码率去卡。

同一张无噪声 PNG, 两把尺子可以给出 2/9 和 9/9。**尺子本身必须先自检**, 否则测出来
的是尺子的锅。这就是本目录的第一条规矩:

> 不过自检 (灰底 8/8 全解回 + 死输入 3/3 全拒), 一律不读 n1..n4 的数。

## 自检 (必须先过)

```bash
python tools/decode/decode_cross_tier.py --selftest
```

```
自检通过 8/8          # 8 个已知 wm_id 的灰底合成图, 必须解回同一个 id
幻觉通道已封 3/3      # 纯平 / 纯噪声 必须被拒, 不能幻觉出 id=0
```

为什么"全零输入解出 id=0"是危险的: **全零码字是合法的 RS(15,5) 码字**,
`decode([0]*15)` 会真的返回 5 个 0 字节, 然后被累加成 wm_id=0。所以拒识门要挡的是
**零证据** (top1−top2 的间隔为 0), 不是挡 id=0。永远不要把 erasure 截到 RS 预算
10 以内再去硬解 —— 那等于把幻觉通道重新打开。

## 布局事实 (核对过的)

```
画布 1920x1080, 块 4行x6列 = 24 块, 每块 320x270
每块 2x2 个码字格, 每格 160x135            ->  96 个码字格
wm_seq 16 个值 (15 个 RS 码字倒序 + 1 个定位索引) 切成 4 组 x 4
组 k 铺在 k = (j + (i%2)*2) % 4 的 6 个块里 ->  16 个符号, 每个 6 路冗余
slot 15 = 回字形定位图案, 位置恰是 get_locator_positions() 那 6 个
```

位尺寸是 **20 x 17 px**: `FIX_FG_MATRIX` 每码字 64 位 = 8x8,
`gen_wm_block(block_size=64)` -> 512x512 -> NEAREST 缩到 160x135,
160/8 = **20 px** x 135/8 ≈ **17 px**。
(160/(512/8) = 2.5 是错的, 那是拿模板空间的 px/位 当位数了。)

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DX_OVERLAY_REPO` | 脚本所在目录的上两级 | 仓库根, 用来找 `watermark_locator/` |
| `DX_DATASET_ROOT` | `/data1/lpl/datasets_labeled_4tier` | 跨档数据集 (clean/n1..n4) |
| `DX_OUT_DIR` | `.` (当前目录) | 结果 JSON 落这里。**建议显式指定**, 免得写进仓库 |

```bash
DX_OUT_DIR=/data1/tmp_vis python tools/decode/decode_cross_tier.py
```

## 脚本索引

### 核心仪器

| 脚本 | 做什么 |
|---|---|
| `decode_cross_tier.py` | 匹配滤波器本体 + 跨档 (clean/n1..n4) 自洽解码。**别的脚本都 import 它** |
| `detect_eval.py` | `detect_at` / `eval_boxes` / `iou` / `nms`。选权重时跨 checkpoint 共用的检测口径 |
| `analyze_vote_gate.py` | 只读 `cross_tier_decode.json`, 扫 `slot_ok >= k` 找拒识门的操作点 |

### 实拍链

| 脚本 | 做什么 | 结论 |
|---|---|---|
| `decode_real_capture.py` | 用本仪器独立解拍前 PNG / 拍后 rect, 顺带量交付振幅 | 拍前 **9/9**, 拍后 **5/18** 落已知集 |
| `decode_polarity.py` | 交换 B/G 让 G−B 反号, 看是不是通道序翻了 | 反极性 0/18 —— 不是极性问题 |
| `decode_wbcorr.py` | 用 4 个黑白回字角标当中性灰参考, 拟合扣掉 `G−B = a + c·L` | slot_ok 抬高 (8→11 / 10→14), 但 13 张死图一张没救回 |
| `decode_envelope.py` | 条纹梳参考 vs 位块包络参考 A/B | 包络不救拍后 (仍 5/18 且同几条), 反把拍前 9/9 打到 6/9 |
| `decode_multiframe.py` | 2 帧匹配滤波分数级融合 (不配准, 对配准误差免疫) | 3/9 对, 仍是那 3 个 ID, slot_ok 反被稀释 (10→8) |
| `ckpt_select_real.py` | 按**实拍召回**选权重, 不用 val fitness | 见下 |

### RS 接口探针

| 脚本 | 做什么 |
|---|---|
| `dbg_rs_order.py` | 查 RS 码字顺序 (不读图) |
| `dbg_rs_erasure.py` | 查 reedsolo 的 erasure 容忍度, 验 `2e + f <= 10` (不读图) |

## 拒识门

```
slot_ok = 6 路投票里 >= VOTE_MIN(4) 路同向的槽数
slot_ok >= 3  才采信
```

`slot_ok >= 3` 是扫出来的最小门, 让**解错 = 0** 同时干净锚存活 60/60 (k<=13)。
解错是自信的错 ID, 比解不出危险得多, 所以门的首要目标是把解错压到 0, 解码率让路。

RS 侧的前置条件 (不是截断):
```python
if len(erase_pos) <= 10:      # 2e + f <= 10, 实测 e=5 OK / e=6 FAIL
    dec = _rs_codec.decode(rs_cw, erase_pos=erase_pos or None)[0]
```

## 选权重: `ckpt_select_real.py`

```bash
python tools/decode/ckpt_select_real.py <weights_dir> [--live-only] [--conf 0.25]
```

**不要按 val fitness 选 best.pt。** v4 的 best.pt 就是按 val 选的, 接上 v3 后灵敏度
一路掉, ep50 反而是最好点, 最终 best 却不是它 —— 选错指标等于白训。部署点是实拍,
就按实拍召回排。

口径 (`detect_eval.py`, 与 `draw_boxes.py` 严格一致, 改了就等于改掉所有历史对比):
```
输入  vis/real_capture/detect_out/rect/*.png   18 张矫正图, 1920x1080
缩放  max(w,h) -> imgsz, INTER_AREA
pad   右/下补到 32 的倍数, 填 114 (YOLO 默认灰)
推理  model.predict(imgsz=max(pw,ph), conf=0.25)
映射  预测框 / s 回原图坐标
GT    6 个固定定位块 160x135
```

`--live-only` 只统计水印残余 > 0 的 6 张 (信号真在的那几张), 用来把"检测器没检出"
和"信号本来就没了"分开。

## 依赖

`numpy` `opencv-python` `reedsolo` `ultralytics` (只有 `ckpt_select_real.py` 要)。
解码器本身不碰 torch, 不需要 GPU。
