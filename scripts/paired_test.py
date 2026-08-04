"""So sánh hai (nhóm) run bằng kiểm định CẶP trên AP từng query.

Vì sao không so hai giá trị mAP trung bình: sai số chuẩn của mAP trên 12,694
query cỡ 0.2 pp, ngang với chênh lệch điển hình giữa hai cấu hình. Nhưng hai
model được đánh giá trên CÙNG tập query, theo cùng thứ tự, và tương quan rất
cao (chung backbone đóng băng, chung residual mix). So theo cặp từng query khử
được phần phương sai chung đó và nhạy hơn nhiều lần.

Đầu vào là các file ap_best.npz do quá trình train ghi vào <run_dir>.

Chạy:
    # so 1 run với 1 run
    python scripts/paired_test.py A/ap_best.npz B/ap_best.npz

    # so 2 nhóm nhiều seed (lấy trung bình AP theo từng query trong mỗi nhóm)
    python scripts/paired_test.py --a run1/ap_best.npz run2/ap_best.npz \
                                  --b run3/ap_best.npz run4/ap_best.npz

    # tự tìm theo glob
    python scripts/paired_test.py --a "tb_logs/base_s*/*/ap_best.npz" \
                                  --b "tb_logs/xchg_s*/*/ap_best.npz"
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace') if hasattr(sys.stdout, 'reconfigure') else None


def load_group(paths, name):
    """Đọc nhiều file, kiểm tra khớp thứ tự query, trả ma trận [n_run, n_query]."""
    if not paths:
        raise SystemExit(f'Nhóm {name}: không có file nào')

    aps, labels_ref, metas = [], None, []
    for p in paths:
        d = np.load(p)
        ap, lab = d['ap'], d['sketch_labels']
        if labels_ref is None:
            labels_ref = lab
        elif not np.array_equal(labels_ref, lab):
            raise SystemExit(
                f'Thứ tự query khác nhau giữa các file (so {paths[0]} với {p}).\n'
                'Kiểm cặp chỉ hợp lệ khi mọi run dùng cùng val loader với shuffle=False.')
        aps.append(ap)
        metas.append((os.path.dirname(p), float(d['best_map']), int(d['epoch'])))

    print(f'Nhóm {name}: {len(aps)} run, {len(aps[0]):,} query')
    for d_, m, e in metas:
        print(f'   mAP={m:.4f} @epoch {e:<3} {d_}')
    return np.stack(aps), labels_ref


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def report(a, b, labels, label_a, label_b, n_boot):
    # Trung bình theo từng query trước, rồi mới so cặp -> khử nhiễu seed
    a_q, b_q = a.mean(axis=0), b.mean(axis=0)
    diff = a_q - b_q
    n = len(diff)

    map_a, map_b = a_q.mean(), b_q.mean()
    se_unpaired = math.sqrt(a_q.var(ddof=1) / n + b_q.var(ddof=1) / n)
    se_paired = diff.std(ddof=1) / math.sqrt(n)
    t = diff.mean() / se_paired if se_paired > 0 else float('inf')
    lo, hi = bootstrap_ci(diff, n_boot=n_boot)
    r = np.corrcoef(a_q, b_q)[0, 1]

    print(f'\n{"=" * 66}')
    print(f'{label_a:>28} mAP = {100 * map_a:.3f}')
    print(f'{label_b:>28} mAP = {100 * map_b:.3f}')
    print(f'{"chênh lệch":>28}     = {100 * diff.mean():+.3f} pp')
    print(f'{"=" * 66}')
    print(f'  tương quan AP giữa 2 nhóm     r = {r:.4f}')
    print(f'  SE nếu so KHÔNG cặp             = {100 * se_unpaired:.3f} pp')
    print(f'  SE khi so CẶP                   = {100 * se_paired:.3f} pp'
          f'   ({se_unpaired / se_paired:.1f}x nhạy hơn)')
    print(f'  t (paired)                      = {t:+.2f}')
    print(f'  KTC bootstrap 95%               = [{100 * lo:+.3f}, {100 * hi:+.3f}] pp')

    n_better = int((diff > 0).sum())
    n_worse = int((diff < 0).sum())
    print(f'  query tốt hơn / kém hơn / hoà   = {n_better:,} / {n_worse:,} / '
          f'{n - n_better - n_worse:,}')

    print()
    significant = lo > 0 or hi < 0
    if significant:
        winner = label_a if diff.mean() > 0 else label_b
        print(f'  KẾT LUẬN: khác biệt CÓ ý nghĩa (KTC không chứa 0). {winner} tốt hơn.')
    else:
        print('  KẾT LUẬN: KHÔNG phân biệt được (KTC 95% chứa 0).')
        print(f'  δ_min hiện tại ≈ {100 * max(abs(lo), abs(hi)):.3f} pp — chênh lệch nhỏ hơn')
        print('  mức này thì dữ liệu hiện có không kết luận được; cần thêm seed.')
    return 0 if significant else 2


def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else ([p] if os.path.exists(p) else []))
    return out


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument('pos', nargs='*', help='hai đường dẫn ap_best.npz (dạng rút gọn)')
    ap_.add_argument('--a', nargs='+', default=None, help='nhóm A (glob được)')
    ap_.add_argument('--b', nargs='+', default=None, help='nhóm B (glob được)')
    ap_.add_argument('--name-a', default='A')
    ap_.add_argument('--name-b', default='B')
    ap_.add_argument('--n-boot', type=int, default=10000)
    args = ap_.parse_args()

    if args.a and args.b:
        pa, pb = expand(args.a), expand(args.b)
    elif len(args.pos) == 2:
        pa, pb = expand([args.pos[0]]), expand([args.pos[1]])
    else:
        ap_.error('Cần --a/--b hoặc đúng hai đường dẫn')

    a, lab_a = load_group(pa, args.name_a)
    b, lab_b = load_group(pb, args.name_b)
    if not np.array_equal(lab_a, lab_b):
        raise SystemExit('Hai nhóm có thứ tự query khác nhau — không kiểm cặp được.')

    return report(a, b, lab_a, args.name_a, args.name_b, args.n_boot)


if __name__ == '__main__':
    sys.exit(main())
