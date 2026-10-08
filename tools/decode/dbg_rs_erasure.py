"""查 reedsolo 的 erasure 接口 —— 只用 _rs_codec, 不读图。"""
import inspect
import os
import sys

REPO = os.environ.get('DX_OVERLAY_REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))

from utils import _rs_codec, rs_encode  # noqa: E402


def main():
    print('decode 签名:', inspect.signature(_rs_codec.decode))
    print('reedsolo 版本:', __import__('reedsolo').__version__ if hasattr(__import__('reedsolo'), '__version__') else '?')

    wid = 0x12345
    seq = rs_encode(wid)
    good = list(seq[:15][::-1])          # 与 decode_cross_tier.py 同一顺序
    print(f'\n正确码字 rs_cw = {good}')

    # 1) 全零码字是不是合法 RS 码字 (幻觉 ID 通道)
    try:
        d = _rs_codec.decode([0] * 15)
        print(f'\n[幻觉通道] decode([0]*15) -> {d[0]!r}   **合法! 会解出 id**')
    except Exception as e:
        print(f'\n[幻觉通道] decode([0]*15) -> FAIL {type(e).__name__}: {e}')

    # 2) 纯 erasure 能容忍几个
    print('\n纯 erasure 容忍度 (码字打成全 0, 位置当 erasure):')
    for f in (4, 5, 6, 8, 10, 11, 15):
        cw = list(good)
        pos = list(range(f))
        for p in pos:
            cw[p] = 0
        try:
            d = _rs_codec.decode(cw, erase_pos=pos)
            got = list(d[0])
            ok = got == [1, 2, 3, 4, 5]
            print(f'  f={f:2d} -> {got} {"OK" if ok else "**错**"}')
        except Exception as e:
            print(f'  f={f:2d} -> FAIL {type(e).__name__}: {e}')

    # 3) 混合: e 个错误 + f 个 erasure, 2e+f <= 10?
    print('\n混合 e 错 + f erasure (2e+f 手算 vs 实测):')
    for e, f in ((1, 0), (2, 0), (5, 0), (6, 0), (0, 10), (1, 8), (2, 6), (3, 4), (4, 2), (5, 0)):
        cw = list(good)
        for i in range(e):
            cw[i] = (cw[i] + 1) % 16      # 制造错误
        # erasure 放码字尾部, 避开开头的 e 个错误位; 测试里 e+f <= 10 < 15 不会撞
        pos = list(range(15 - f, 15))
        assert not set(pos) & set(range(e)), 'erasure 与 error 位重叠'
        for p in pos:
            cw[p] = 0
        try:
            d = _rs_codec.decode(cw, erase_pos=pos if pos else None)
            got = list(d[0])
            ok = got == [1, 2, 3, 4, 5]
            print(f'  e={e} f={f}  2e+f={2*e+f:2d}  -> {"OK" if ok else f"解出 {got} **错**"}')
        except Exception as e2:
            print(f'  e={e} f={f}  2e+f={2*e+f:2d}  -> FAIL {type(e2).__name__}: {e2}')


if __name__ == '__main__':
    main()
