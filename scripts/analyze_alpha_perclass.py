"""Phân tích theo LỚP: nhánh frozen giúp ở đâu, và giúp bao nhiêu.

Vì sao cần bước này trước khi thiết kế bất kỳ regularizer nào:

Kiểm cặp α=0.5 vs α=0.6 cho `4,639 query tốt hơn / 4,977 kém hơn` — nhiều query
bị α=0.5 làm TỆ đi hơn là làm tốt lên, vậy mà trung bình vẫn cao hơn. Lợi thế
đến từ **thắng đậm trên một nhóm nhỏ**, không phải cải thiện đều tay.

Nếu nhóm đó tập trung ở vài lớp cụ thể thì ta biết nhánh frozen đang bù đắp
chính xác cái gì — và đó là thông tin bắt buộc để thiết kế đúng cách giữ lại nó.
Nếu nó rải đều mọi lớp thì giả thuyết "frozen bù cho một loại nội dung cụ thể"
bị bác bỏ.

Chỉ cần numpy — đọc các file ap_alpha*.npz do sweep_alpha.py --save_ap ghi ra.

Chạy:
    python scripts/analyze_alpha_perclass.py --run_dir tb_logs/p4_alphasweep_s1/2026...
    python scripts/analyze_alpha_perclass.py --run_dir ... --dataset sketchy_ext
"""

import argparse
import ast
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def class_names(dataset):
    """Đọc UNSEEN_CLASSES bằng AST — không import src.dataset_retrieval (cần torch).

    ValidDataset dùng `sorted(set(unseen_classes))`, nên nhãn số chính là chỉ số
    trong danh sách đã sort. Phải khớp đúng thứ tự đó.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'dataset_retrieval.py'), encoding='utf-8').read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'UNSEEN_CLASSES':
            d = ast.literal_eval(node.value)
            return sorted(set(d.get(dataset, d['sketchy'])))
    raise SystemExit('Không đọc được UNSEEN_CLASSES')


def load_sweep(run_dir):
    out, labels = {}, None
    for p in sorted(glob.glob(os.path.join(run_dir, 'ap_alpha*.npz'))):
        m = re.search(r'ap_alpha([0-9.]+)\.npz$', p)
        if not m:
            continue
        d = np.load(p)
        a = float(m.group(1))
        out[a] = d['ap']
        if labels is None:
            labels = d['sketch_labels']
        elif not np.array_equal(labels, d['sketch_labels']):
            raise SystemExit(f'{p}: thứ tự query khác các file kia')
    if not out:
        raise SystemExit(f'Không thấy ap_alpha*.npz trong {run_dir}. '
                         'Chạy sweep_alpha.py với --save_ap trước.')
    return out, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--dataset', default='sketchy_ext')
    ap.add_argument('--ref', type=float, default=1.0, help='α của nhánh prompted thuần')
    ap.add_argument('--best', type=float, default=0.5, help='α của hỗn hợp tốt nhất')
    ap.add_argument('--bins', type=int, default=10,
                    help='số nhóm phân vị khi đo phần trần với tới được bằng tín hiệu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sweep, labels = load_sweep(args.run_dir)
    names = class_names(args.dataset)
    alphas = sorted(sweep)
    print(f'{len(alphas)} giá trị α: {alphas}')
    print(f'{len(labels):,} query, {len(set(labels.tolist()))} lớp\n')

    for a in (args.ref, args.best):
        if a not in sweep:
            raise SystemExit(f'Thiếu α={a} trong dữ liệu (có: {alphas})')

    ref, best = sweep[args.ref], sweep[args.best]
    gain = best - ref            # hỗn hợp giúp bao nhiêu so với prompted thuần

    # ---- 1. Phân bố mức giúp trên từng query ----
    print('=== Nhánh frozen giúp ĐỀU hay TẬP TRUNG? ===')
    q = np.percentile(gain, [1, 10, 25, 50, 75, 90, 99]) * 100
    print(f'  phân vị gain (pp): 1%={q[0]:+.1f}  10%={q[1]:+.1f}  25%={q[2]:+.1f}  '
          f'50%={q[3]:+.1f}  75%={q[4]:+.1f}  90%={q[5]:+.1f}  99%={q[6]:+.1f}')
    print(f'  query được giúp: {100 * (gain > 0).mean():.1f}%   '
          f'bị hại: {100 * (gain < 0).mean():.1f}%   không đổi: {100 * (gain == 0).mean():.1f}%')

    # Chia riêng phần dương và phần âm. Dùng "tỉ lệ trên tổng RÒNG" sẽ vượt 100%
    # khi có nhiều query bị hại, và con số đó không diễn giải được.
    order = np.argsort(-gain)
    tot_pos, tot_neg = gain[gain > 0].sum(), gain[gain < 0].sum()
    print(f'  tổng phần được giúp {100 * tot_pos / len(gain):+.2f} pp, '
          f'phần bị hại {100 * tot_neg / len(gain):+.2f} pp, '
          f'ròng {100 * gain.mean():+.2f} pp')
    for frac in (0.05, 0.10, 0.25):
        k = int(len(gain) * frac)
        share = gain[order[:k]].sum() / tot_pos if tot_pos != 0 else float('nan')
        print(f'  {frac:.0%} query tốt nhất chiếm {100 * share:.0f}% TỔNG PHẦN ĐƯỢC GIÚP '
              f'(đều tay = {100 * frac:.0f}%)')

    # ---- 2. Theo lớp ----
    print(f'\n=== Theo lớp: α={args.best} so với α={args.ref} (prompted thuần) ===')
    print(f'{"lớp":<16} {"n":>5} {"prompted":>9} {"hỗn hợp":>9} {"gain":>8}  {"α tốt nhất":>10}')
    print('-' * 64)
    print(f'{"":<16} {"":>5} {"":>9} {"":>9} {"±SE":>8}')
    rows = []
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        per_alpha = {a: sweep[a][m].mean() for a in alphas}
        a_star = max(per_alpha, key=per_alpha.get)
        gi = gain[m]
        rows.append({
            'class': names[lab] if lab < len(names) else f'#{lab}',
            'n': int(m.sum()),
            'prompted': ref[m].mean(), 'mixed': best[m].mean(),
            'gain': gi.mean(),
            # SE của hiệu THEO CẶP trong lớp — nhỏ hơn nhiều so với SE của mAP,
            # vì cùng query nên phần khó/dễ của query bị khử.
            'gain_se': gi.std(ddof=1) / np.sqrt(len(gi)),
            'alpha_star': a_star,
        })
    for r in sorted(rows, key=lambda r: -r['gain']):
        sig = '*' if abs(r['gain']) > 2 * r['gain_se'] else ' '
        print(f'{r["class"]:<16} {r["n"]:>5} {100 * r["prompted"]:>9.2f} '
              f'{100 * r["mixed"]:>9.2f} {100 * r["gain"]:>+8.2f}{sig} '
              f'{100 * r["gain_se"]:>6.2f} {r["alpha_star"]:>10.2f}')
    print('  (* = |gain| > 2·SE, tức phân biệt được với 0)')

    g = np.array([r['gain'] for r in rows])
    se = np.array([r['gain_se'] for r in rows])
    astar = np.array([r['alpha_star'] for r in rows])
    med = np.median(g)

    print(f'\n  gain theo lớp: min {100 * g.min():+.2f}, trung vị {100 * med:+.2f}, '
          f'max {100 * g.max():+.2f}')
    print(f'  SE điển hình theo lớp: {100 * np.median(se):.2f} pp')
    n_sig = int((np.abs(g) > 2 * se).sum())
    n_neg = int((g < 0).sum())
    print(f'  {n_sig}/{len(g)} lớp có gain phân biệt được với 0; {n_neg} lớp bị TỆ đi')
    print(f'  α tốt nhất theo lớp: min {astar.min():.2f}, trung vị {np.median(astar):.2f}, '
          f'max {astar.max():.2f}  (std {astar.std():.3f})')

    # ---- Cận trên của α thích ứng theo nội dung ----
    # Nếu chọn được α tối ưu cho TỪNG LỚP thì mAP là bao nhiêu? Đây là oracle:
    # không thực hiện được (cần nhãn tập test), nhưng nó chặn trên mọi phương pháp
    # dự đoán α từ nội dung. Nếu cận trên gần với α toàn cục thì hướng đó vô ích.
    n_tot = len(labels)
    glob_best = max(alphas, key=lambda a: sweep[a].mean())
    oracle = sum(max(sweep[a][labels == r_lab].mean() for a in alphas) * (labels == r_lab).sum()
                 for r_lab in sorted(set(labels.tolist())))/ n_tot
    # Oracle theo TỪNG QUERY: chọn α tốt nhất cho mỗi query riêng lẻ. Đây mới là
    # trần tuyệt đối của mọi cách chọn α. Oracle theo LỚP KHÔNG chặn trên cổng
    # theo query, vì α* còn biến thiên trong nội bộ từng lớp.
    per_query_oracle = float(np.max(np.stack([sweep[a] for a in alphas]), axis=0).mean())
    gm = float(sweep[glob_best].mean())
    print(f'\n=== Các mức oracle của α thích ứng ===')
    print(f'  α toàn cục tốt nhất ({glob_best})    : {100 * gm:.3f}   (đạt được)')
    print(f'  oracle theo LỚP                : {100 * oracle:.3f}   '
          f'({100 * (oracle - gm):+.3f} pp — cần nhãn)')
    print(f'  oracle theo TỪNG QUERY         : {100 * per_query_oracle:.3f}   '
          f'({100 * (per_query_oracle - gm):+.3f} pp — trần tuyệt đối)')

    # ---- Phần trần với tới được bằng tín hiệu quan sát được ----
    # Oracle theo lớp là TRẦN nhưng cần nhãn. Ở đây hỏi câu khác: nếu chỉ dùng một
    # tín hiệu tính được lúc suy luận, chia query theo phân vị của tín hiệu đó rồi
    # chọn α tối ưu cho từng nhóm, thì với tới bao nhiêu phần của trần?
    # Vẫn là oracle (chọn α trên tập test) nhưng bị RÀNG BUỘC chỉ được dùng tín
    # hiệu quan sát được — nên nó chặn trên mọi cổng học từ tín hiệu ấy.
    sig_path = os.path.join(args.run_dir, 'alpha_signals.npz')
    if os.path.exists(sig_path):
        sig = np.load(sig_path)
        glob_map = sweep[glob_best].mean()
        print(f'\n=== Phần trần với tới được bằng tín hiệu quan sát được ===')
        print(f'  (chia {args.bins} nhóm theo phân vị, oracle α cho mỗi nhóm)')
        print(f'  {"tín hiệu":<18} {"mAP":>8} {"so với α toàn cục":>18} {"% trần query":>13}')
        print('  ' + '-' * 62)
        head = per_query_oracle - glob_map
        best_sig = None
        for name in sig.files:
            s = sig[name]
            if len(s) != n_tot:
                continue
            edges = np.quantile(s, np.linspace(0, 1, args.bins + 1))
            edges[-1] += 1e-9
            tot = 0.0
            for i in range(args.bins):
                m = (s >= edges[i]) & (s < edges[i + 1])
                if m.sum() == 0:
                    continue
                tot += max(sweep[a][m].mean() for a in alphas) * m.sum()
            v = tot / n_tot
            frac = (v - glob_map) / head if head > 0 else float('nan')
            print(f'  {name:<18} {100 * v:>8.3f} {100 * (v - glob_map):>+18.3f} '
                  f'{100 * frac:>10.0f}%')
            if best_sig is None or v > best_sig[1]:
                best_sig = (name, v)
        if best_sig:
            print(f'\n  Tín hiệu tốt nhất: {best_sig[0]} -> {100 * best_sig[1]:.3f} '
                  f'({100 * (best_sig[1] - glob_map):+.3f} pp)')
            print('  Đây vẫn là oracle (chọn α trên tập test) nên là TRẦN của mọi cổng học')
            print('  từ tín hiệu đó. Cổng thật sẽ đạt ít hơn.')
    else:
        print(f'\n(Không thấy {sig_path} — chạy lại sweep_alpha.py --save_ap để có')
        print(' phân tích "tín hiệu nào với tới được phần trần".)')

    # ---- Diễn giải: báo từng tín hiệu riêng, không gộp thành một phán quyết ----
    print('\n=== Diễn giải ===')

    # Có tín hiệu nào dự đoán được α* không? Tương quan với chất lượng nhánh
    # prompted là ứng viên đầu tiên và không cần thêm dữ liệu gì.
    pv = np.array([r['prompted'] for r in rows])
    if pv.std() > 0 and astar.std() > 0:
        r_pa = float(np.corrcoef(pv, astar)[0, 1])
        r_pg = float(np.corrcoef(pv, g)[0, 1])
        print(f'  [0] corr(chất lượng prompted, α*) = {r_pa:+.3f} ; '
              f'corr(chất lượng prompted, gain) = {r_pg:+.3f}')
        if abs(r_pa) > 0.3 or abs(r_pg) > 0.3:
            print('      -> α* CÓ cấu trúc dự đoán được: lớp mà nhánh prompted đã tốt thì muốn α')
            print('         cao, lớp prompted yếu thì cần frozen bù. Một cổng α phụ thuộc nội')
            print('         dung là khả thi về nguyên tắc.')
    spread = (g.max() - g.min()) / abs(med) if med != 0 else float('inf')
    print(f'  [1] Độ tản theo lớp = (max−min)/trung vị = {spread:.2f}')
    if spread > 0.5:
        print('      -> KHÔNG đều. Nhánh frozen bù cho một loại nội dung cụ thể; regularizer')
        print('         nên nhắm vào loại đó thay vì áp đều toàn cục.')
    else:
        print('      -> Khá đều. Nhánh frozen đóng vai trò điều chuẩn chung, không đặc thù lớp.')

    k = int(len(gain) * 0.25)
    share = gain[order[:k]].sum() / tot_pos if tot_pos != 0 else float('nan')
    print(f'  [2] 25% query tốt nhất chiếm {100 * share:.0f}% tổng phần được giúp '
          f'(đều tay = 25%)')
    if share > 0.6:
        print('      -> Tập trung mạnh. Lợi ích đến từ thiểu số query, không phải cải thiện đều.')

    if n_neg:
        print(f'  [3] {n_neg} lớp bị hỗn hợp làm tệ đi -> α toàn cục là thoả hiệp, có lớp trả giá.')
    if astar.std() > 0.05:
        print(f'  [4] α tối ưu biến thiên theo lớp (std {astar.std():.3f}) -> α phụ thuộc nội dung')
        print('      có thể còn dư địa; nhưng cẩn thận, chọn α theo lớp trên tập test là gian lận.')

    out = args.out or os.path.join(args.run_dir, 'alpha_perclass.csv')
    import csv
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nCSV: {os.path.abspath(out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
