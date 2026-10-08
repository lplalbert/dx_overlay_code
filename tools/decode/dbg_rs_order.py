"""查 RS 码字顺序 —— 只用 rs_encode / _rs_codec, 不读图。"""
import os
import sys

REPO = os.environ.get('DX_OVERLAY_REPO', os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
sys.path.insert(0, os.path.join(REPO, 'watermark_locator'))

from utils import rs_encode, _rs_codec, LOCATOR_CODEWORD_INDEX  # noqa: E402

WID = 0x12345

def main():
    seq = rs_encode(WID)
    print(f'wm_id=0x{WID:05X}')
    print(f'wm_seq (16) = {seq}')
    print(f'最后一位 = {seq[-1]} (应为 LOCATOR_CODEWORD_INDEX={LOCATOR_CODEWORD_INDEX})')
    print(f'前15位 = {seq[:15]}')

    cands = {
        'A  wm_seq[:15][::-1]': list(seq[:15][::-1]),
        'B  wm_seq[:15]': list(seq[:15]),
        'C  wm_seq[1:16]': list(seq[1:16]),
        'D  wm_seq[1:16][::-1]': list(seq[1:16][::-1]),
    }
    for name, cw in cands.items():
        for as_type in ('list', 'bytearray'):
            arg = bytearray(cw) if as_type == 'bytearray' else list(cw)
            try:
                d = _rs_codec.decode(arg)
                print(f'{name:24s} [{as_type:9s}] -> type={type(d).__name__}  {d}')
            except Exception as e:
                print(f'{name:24s} [{as_type:9s}] -> FAIL {type(e).__name__}: {e}')

    print('\n直接看 encode 输出:')
    data = [0x1, 0x2, 0x3, 0x4, 0x5]
    enc = list(_rs_codec.encode(data))
    print(f'encode({data}) = {enc}  len={len(enc)}')
    d = _rs_codec.decode(enc)
    print(f'decode(enc) = {d!r}')
    if isinstance(d, tuple):
        print(f'  tuple 长度 {len(d)}: 各元素 {[type(x).__name__ for x in d]}')
        print(f'  d[0] = {list(d[0]) if hasattr(d[0], "__iter__") else d[0]}')


if __name__ == '__main__':
    main()
