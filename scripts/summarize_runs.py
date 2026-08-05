"""Tổng hợp runs_summary.csv thành bảng ablation, và phân tích metrics_epoch.csv.

Bảng trong bài báo nên được SINH RA từ dữ liệu chứ không chép tay. Script này
đọc runs_summary.csv (một dòng mỗi run, do RunCSVLogger ghi), gộp các run cùng
cấu hình khác seed, và xuất markdown kèm mean ± std.

Chạy:
    # bảng ablation, tự phát hiện siêu tham số nào thay đổi giữa các run
    python scripts/summarize_runs.py

    # gộp theo cột cụ thể
    python scripts/summarize_runs.py --group-by cfg_n_ctx cfg_disable_cross_exchange

    # phân tích epoch: best epoch nằm đâu, screening ngắn có xếp hạng đúng không
    python scripts/summarize_runs.py --epochs

    # chỉ lấy các run có tên khớp
    python scripts/summarize_runs.py --filter n_ctx
"""

import argparse
import csv
import os
import statistics
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f'Không tìm thấy {path}. Chạy train ít nhất một lần trước.')
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def varying_cfg_cols(rows):
    """Các cột cfg_* thực sự thay đổi giữa các run — chính là trục ablation."""
    cols = [c for c in rows[0] if c.startswith('cfg_')]
    return [c for c in cols if len({r.get(c, '') for r in rows}) > 1]


def fmt_group(vals, group_cols):
    return ', '.join(f'{c[4:]}={v}' for c, v in zip(group_cols, vals))


def ablation_table(rows, group_cols, metric):
    groups = {}
    for r in rows:
        key = tuple(r.get(c, '') for c in group_cols)
        v = to_float(r.get(metric))
        if v is not None:
            groups.setdefault(key, []).append((v, r))

    if not groups:
        raise SystemExit(f'Không dòng nào có cột {metric!r} là số. '
                         'Các run trước bản vá hook on_validation_end sẽ trống cột này.')

    print(f'\n| Cấu hình | n seed | {metric} (mean ± std) | min | max | params |')
    print('|---|---:|---:|---:|---:|---:|')
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: -statistics.fmean(v for v, _ in kv[1])):
        vals = [v for v, _ in items]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        params = items[0][1].get('trainable_params', '')
        params_s = f'{int(params):,}' if params.isdigit() else params
        sd_s = f' ± {100 * sd:.2f}' if len(vals) > 1 else ' ± —'
        print(f'| {fmt_group(key, group_cols)} | {len(vals)} | '
              f'{100 * mean:.2f}{sd_s} | {100 * min(vals):.2f} | {100 * max(vals):.2f} | {params_s} |')
        out.append((key, vals))

    singles = [k for k, v in out if len(v) < 3]
    if singles:
        print(f'\n  ⚠ {len(singles)} cấu hình có < 3 seed — chưa ước lượng được dao động.')
    print('\n  Chênh lệch nhỏ hơn ~2×std là không kết luận được.')
    print('  Dùng scripts/paired_test.py trên ap_best.npz để có kết luận chặt hơn.')
    return out


def epoch_analysis(rows, screen_epoch):
    """Best epoch nằm ở đâu, và screening ngắn có xếp hạng đúng như full không."""
    curves = []
    for r in rows:
        rd = r.get('run_dir', '')
        p = os.path.join(rd, 'metrics_epoch.csv')
        if not rd or not os.path.exists(p):
            continue
        er = read_csv(p)
        pts = [(int(float(e['epoch'])), to_float(e.get('mAP')))
               for e in er if to_float(e.get('mAP')) is not None]
        if pts:
            curves.append((r.get('run_id', '?'), r.get('exp_name', '?'), sorted(pts)))

    if not curves:
        print('\nKhông đọc được metrics_epoch.csv nào (cột run_dir trống hoặc file đã bị xoá).')
        return

    print(f'\n=== Best epoch trên {len(curves)} run ===')
    bests, lasts = [], []
    for rid, exp, pts in curves:
        be, bv = max(pts, key=lambda t: t[1])
        le, lv = pts[-1]
        bests.append(be)
        lasts.append((bv - lv) * 100)
        print(f'  {exp:<28} best @epoch {be:>3} ({100 * bv:.2f})   '
              f'cuối @epoch {le:>3} ({100 * lv:.2f})')

    med = statistics.median(bests)
    print(f'\n  best epoch: median {med:.0f}, khoảng [{min(bests)}, {max(bests)}]')
    if max(bests) < 0.7 * max(p[-1][0] for _, _, p in curves):
        print('  → Hội tụ sớm. Rút ngắn số epoch sẽ tiết kiệm GPU mà không mất gì.')
    gap = statistics.fmean(lasts)
    if gap > 0.3:
        print(f'  → Epoch cuối kém best trung bình {gap:.2f} pp: BẮT BUỘC dùng best-epoch, '
              'không dùng epoch cuối.')

    # Screening ngắn có xếp hạng đúng không?
    # Phép này chỉ có nghĩa khi so các CẤU HÌNH KHÁC NHAU. Nếu mọi run chỉ khác
    # seed thì "xếp hạng" là xếp hạng nhiễu, và nếu screen_epoch đã bao trùm
    # best epoch thì rho = 1.0 theo định nghĩa - vô nghĩa hoàn toàn.
    max_epoch = max(p[-1][0] for _, _, p in curves)
    if screen_epoch >= max_epoch:
        print(f'\n=== Screening: BỎ QUA ===')
        print(f'  screen_epoch={screen_epoch} >= epoch lớn nhất đã chạy ({max_epoch}).')
        print('  Cửa sổ screening bao trùm cả best epoch nên rho = 1.0 theo định nghĩa.')
        print(f'  Chạy lại với --screen-epoch nhỏ hơn best epoch median '
              f'({statistics.median(bests):.0f}) mới có thông tin.')
        return

    usable = [(exp, dict(pts)) for _, exp, pts in curves]
    early, final = [], []
    for exp, d in usable:
        e = max((v for k, v in d.items() if k <= screen_epoch), default=None)
        f = max(d.values())
        if e is not None:
            early.append((exp, e))
            final.append((exp, f))
    if len(early) >= 3:
        re_ = {e: i for i, (e, _) in enumerate(sorted(early, key=lambda t: -t[1]))}
        rf = {e: i for i, (e, _) in enumerate(sorted(final, key=lambda t: -t[1]))}
        common = set(re_) & set(rf)
        d2 = sum((re_[k] - rf[k]) ** 2 for k in common)
        n = len(common)
        rho = 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float('nan')
        print(f'\n=== Screening @epoch ≤{screen_epoch} vs kết quả cuối ===')
        print(f'  Spearman ρ = {rho:.3f} trên {n} run')
        if rho >= 0.9:
            print(f'  → Screening {screen_epoch} epoch xếp hạng đáng tin. '
                  'Dùng nó để sàng lọc, tiết kiệm phần lớn GPU.')
        else:
            print(f'  → Screening {screen_epoch} epoch KHÔNG xếp hạng đáng tin. '
                  'Phải chạy đủ epoch cho mọi cấu hình.')
    else:
        print(f'\n(Cần ≥3 run để đánh giá độ tin cậy của screening; hiện có {len(early)}.)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', default='runs_summary.csv')
    ap.add_argument('--group-by', nargs='+', default=None,
                    help='cột gộp; mặc định tự phát hiện cfg_* nào thay đổi')
    ap.add_argument('--metric', default='best_value')
    ap.add_argument('--filter', default=None, help='chỉ giữ run có exp_name chứa chuỗi này')
    ap.add_argument('--epochs', action='store_true', help='phân tích metrics_epoch.csv')
    ap.add_argument('--screen-epoch', type=int, default=20)
    args = ap.parse_args()

    rows = read_csv(args.summary)
    rows = [r for r in rows if r.get('status') == 'completed'] or rows
    if args.filter:
        rows = [r for r in rows if args.filter in r.get('exp_name', '')]
    if not rows:
        raise SystemExit('Không còn run nào sau khi lọc.')

    print(f'{len(rows)} run trong {args.summary}')

    group_cols = args.group_by or varying_cfg_cols(rows)
    same_config = not group_cols
    if same_config:
        group_cols = ['exp_name']
        print('(Mọi cfg_* giống nhau — đây là các run CHỈ KHÁC SEED, không phải ablation.')
        print(' Bảng dưới đo dao động giữa các seed, không so sánh phương pháp.)')
    else:
        print(f'Trục ablation: {", ".join(c[4:] for c in group_cols)}')

    ablation_table(rows, group_cols, args.metric)

    if same_config:
        vals = [to_float(r.get(args.metric)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            print(f'\n=== Dao động giữa seed ===')
            print(f'  n={len(vals)}, mean {100 * statistics.fmean(vals):.2f}, '
                  f'std {100 * sd:.2f} pp, khoảng {100 * (max(vals) - min(vals)):.2f} pp')
            print(f'  → Mọi chênh lệch nhỏ hơn ~{100 * 2 * sd:.2f} pp là nhiễu seed.')

    if args.epochs:
        epoch_analysis(rows, args.screen_epoch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
