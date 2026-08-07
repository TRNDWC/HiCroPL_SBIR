"""Cải tiến phía đánh giá, KHÔNG train lại: α theo cụm (H′) và hậu xử lý (F).

Cả hai chạy trên đặc trưng trích từ một checkpoint đã có, nên một lần trích dùng
cho mọi thí nghiệm.

H′ — α theo CỤM. Cổng α theo từng query đã thất bại (tín hiệu tốt nhất với tới
3% dư địa), NHƯNG cấu trúc mức lớp là thật (+2.868 pp, sống sót kiểm chứng chia
đôi). Vấn đề là không đọc được cấu trúc lớp từ một query riêng lẻ. ZS-SBIR lại
cho toàn bộ gallery lúc test, nên phục hồi cấu trúc đó bằng phân cụm không giám
sát rồi gán α theo cụm. Transductive — hợp lệ trong retrieval, phải báo cáo rõ.
Luôn kèm kiểm chứng chia đôi: chọn α trên nửa query A, chấm trên nửa B.

F — hậu xử lý. αQE (mở rộng truy vấn) và DBA (làm giàu gallery) là hai thủ thuật
chuẩn trong retrieval, không cần train, thường +1-3 mAP. Đây là hướng duy nhất
còn lại KHÔNG đi ngược mẫu hình đã đo ("tăng thích nghi = tệ hơn").

Chạy:
    python scripts/improve_eval.py --run_dir tb_logs/p4_alphasweep_s1/2026...
    python scripts/improve_eval.py --run_dir ... --clusters 10 21 42 --qe_k 1 2 3 5
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from scripts.sweep_alpha import build, extract, find_ckpt, load_cfg, load_ckpt, mix  # noqa: E402


def l2(x):
    return x / x.norm(dim=-1, keepdim=True)


def kmeans(x, k, iters=50, seed=0):
    """k-means đơn giản trên GPU — tránh phụ thuộc sklearn."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        assign = (x @ c.t()).argmax(1)
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = l2(x[m].mean(0))
    return l2(c)


def alpha_qe(q, g, sim, k, power):
    """αQE: truy vấn mới = q + Σ w_i·g_i, w_i = sim_i^power trên top-k láng giềng."""
    if k <= 0:
        return q
    v, idx = sim.topk(k, dim=-1)
    w = v.clamp(min=0) ** power
    return l2(q + (w.unsqueeze(-1) * g[idx]).sum(1))


def dba(g, k, power):
    """DBA: mỗi vector gallery được làm giàu bằng láng giềng của chính nó."""
    if k <= 0:
        return g
    s = g @ g.t()
    s.fill_diagonal_(-2)
    v, idx = s.topk(k, dim=-1)
    w = v.clamp(min=0) ** power
    out = l2(g + (w.unsqueeze(-1) * g[idx]).sum(1))
    del s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--alpha', type=float, default=0.5, help='α nền để so sánh')
    ap.add_argument('--alphas', nargs='+', type=float,
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
                    help='dải α cho H′')
    ap.add_argument('--clusters', nargs='+', type=int, default=[21, 42, 84, 168])
    ap.add_argument('--qe_k', nargs='+', type=int, default=[0, 5, 10, 20, 40, 75, 120])
    ap.add_argument('--qe_power', type=float, default=3.0)
    ap.add_argument('--dba_k', nargs='+', type=int, default=[0])
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--device', default='cuda' if (torch and torch.cuda.is_available()) else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if torch is None:
        print('Cần cài torch.', file=sys.stderr)
        return 1

    from src.model_hicropl import retrieval_metrics, retrieval_topk

    cfg = load_cfg(args.run_dir, {'data_dir': args.data_dir,
                                  'test_batch_size': None, 'workers': None})
    ckpt = find_ckpt(cfg, args.ckpt)
    print(f'ckpt: {ckpt}\ndataset: {cfg.dataset} | device: {args.device}\n')

    from src.dataset_retrieval import ValidDataset
    lit = build(cfg, args.device)
    load_ckpt(lit, ckpt)
    mk = lambda m: DataLoader(ValidDataset(cfg, mode=m), batch_size=cfg.test_batch_size,
                              num_workers=cfg.workers, shuffle=False)
    print('\nTrích đặc trưng (một lần cho mọi thí nghiệm):')
    uq, fq, lq = extract(lit, mk('sketch'), 'sketch', args.device)
    ug, fg, lg = extract(lit, mk('photo'), 'photo', args.device)
    del lit
    torch.cuda.empty_cache() if args.device == 'cuda' else None

    dev = args.device
    uq, fq, ug, fg = uq.to(dev), fq.to(dev), ug.to(dev), fg.to(dev)
    lq, lg = lq.to(dev), lg.to(dev)
    map_k, p_k = retrieval_topk(cfg.dataset)

    def score(q, g):
        m, _, apv, _, _, _ = retrieval_metrics(q, g, lq, lg, cfg.dataset)
        return float(m), apv

    q0, g0 = mix(uq, fq, args.alpha), mix(ug, fg, args.alpha)
    base, base_ap = score(q0, g0)
    print(f'\n{"=" * 62}\nNỀN: α={args.alpha} -> mAP@{map_k or "all"} = {100 * base:.3f}\n{"=" * 62}')

    rows = [{'method': f'baseline α={args.alpha}', 'param': '', 'mAP': base, 'delta': 0.0}]

    # ---------------- H′ : α theo cụm, có kiểm chứng chia đôi ----------------
    print('\n### H′ — α theo cụm (phân cụm KHÔNG giám sát trên gallery)')
    print('Chọn α trên nửa query A, chấm trên nửa B. Chỉ dòng "giữ lại" mới đáng tin.')
    print(f'{"#cụm":>6} {"oracle":>9} {"giữ lại":>9} {"so với nền":>11}')
    print('-' * 40)
    rs = np.random.default_rng(0)
    best_cluster = (0, -1.0)
    half = torch.from_numpy(rs.random(len(lq)) < 0.5).to(dev)
    ap_by_alpha = {a: score(mix(uq, fq, a), mix(ug, fg, a))[1] for a in args.alphas}

    for k in args.clusters:
        cent = kmeans(g0, k, seed=0)
        assign = (q0 @ cent.t()).argmax(1)          # query -> cụm gần nhất
        orc = held = 0.0
        nB = int((~half).sum())
        for j in range(k):
            m = assign == j
            if not m.any():
                continue
            orc += max(ap_by_alpha[a][m].sum().item() for a in args.alphas)
            mA, mB = m & half, m & ~half
            if mA.sum() == 0 or mB.sum() == 0:
                continue
            aj = max(args.alphas, key=lambda a: ap_by_alpha[a][mA].mean().item())
            held += ap_by_alpha[aj][mB].sum().item()
        orc /= len(lq)
        held /= max(nB, 1)
        base_B = base_ap[~half].mean().item()
        print(f'{k:>6} {100 * orc:>9.3f} {100 * held:>9.3f} {100 * (held - base_B):>+11.3f}')
        rows.append({'method': 'H′ cluster-α (giữ lại)', 'param': f'k={k}',
                     'mAP': held, 'delta': held - base_B})
        if held > best_cluster[1]:
            best_cluster = (k, held)

    # ---------------- F : hậu xử lý ----------------
    print('\n### F — hậu xử lý (αQE trên truy vấn, DBA trên gallery)')
    print(f'{"qe_k":>5} {"dba_k":>6} {"mAP":>9} {"so với nền":>11}')
    print('-' * 34)
    best = (base, 0, 0)
    for dk in args.dba_k:
        gA = dba(g0, dk, args.qe_power)
        sim = q0 @ gA.t()
        for qk in args.qe_k:
            qA = alpha_qe(q0, gA, sim, qk, args.qe_power)
            v, _ = score(qA, gA)
            flag = ' <-' if v > best[0] else ''
            print(f'{qk:>5} {dk:>6} {100 * v:>9.3f} {100 * (v - base):>+11.3f}{flag}')
            rows.append({'method': 'F postproc', 'param': f'qe_k={qk},dba_k={dk}',
                         'mAP': v, 'delta': v - base})
            if v > best[0]:
                best = (v, qk, dk)
        del gA, sim

    print(f'\nTốt nhất: qe_k={best[1]}, dba_k={best[2]} -> {100 * best[0]:.3f} '
          f'({100 * (best[0] - base):+.3f} pp so với nền {100 * base:.3f})')

    # ---------------- Kết hợp: α theo cụm rồi αQE ----------------
    # Hai cải tiến tác động lên hai trục khác nhau (α đổi đặc trưng, QE đổi truy
    # vấn), nên câu hỏi là chúng có CỘNG DỒN hay giẫm chân nhau. Chấm trên nửa B
    # với α chọn từ nửa A, để con số so được với cột "giữ lại" của H′.
    if best[1] > 0 and best_cluster[0] > 0:
        kbest = best_cluster[0]   # k tốt nhất theo cột giữ lại, KHÔNG phải k lớn nhất
        cent = kmeans(g0, kbest, seed=0)
        assign = (q0 @ cent.t()).argmax(1)
        qmix = q0.clone()
        for j in range(kbest):
            m = assign == j
            mA = m & half
            if not m.any() or mA.sum() == 0:
                continue
            aj = max(args.alphas, key=lambda a: ap_by_alpha[a][mA].mean().item())
            qmix[m] = mix(uq[m], fq[m], aj)
        B = ~half
        base_B = base_ap[B].mean().item()

        def score_B(qq, gg):
            m_, _, apv, _, _, _ = retrieval_metrics(qq, gg, lq, lg, cfg.dataset)
            return apv[B].mean().item()

        gA = dba(g0, best[2], args.qe_power)
        simc = qmix @ gA.t()
        v_comb = score_B(alpha_qe(qmix, gA, simc, best[1], args.qe_power), gA)
        v_qe_only = score_B(alpha_qe(q0, gA, q0 @ gA.t(), best[1], args.qe_power), gA)
        v_cl_only = score_B(qmix, g0)

        print(f'\n### Kết hợp (chấm trên nửa B, α cụm chọn từ nửa A, k={kbest})')
        print(f'  nền                       {100 * base_B:>8.3f}')
        print(f'  chỉ α theo cụm            {100 * v_cl_only:>8.3f}  {100 * (v_cl_only - base_B):+.3f}')
        print(f'  chỉ αQE (qe_k={best[1]})  {100 * v_qe_only:>8.3f}  {100 * (v_qe_only - base_B):+.3f}')
        print(f'  CẢ HAI                    {100 * v_comb:>8.3f}  {100 * (v_comb - base_B):+.3f}')
        add = (v_comb - base_B) - (v_cl_only - base_B) - (v_qe_only - base_B)
        print(f'  -> {"cộng dồn tốt" if add > -0.002 else "giẫm chân nhau"} '
              f'(chênh so với tổng hai phần riêng: {100 * add:+.3f} pp)')
        rows.append({'method': 'H′+F kết hợp', 'param': f'k={kbest},qe_k={best[1]}',
                     'mAP': v_comb, 'delta': v_comb - base_B})
        del gA, simc

    # ---------------- H″ : α theo cụm KHÔNG dùng nhãn ----------------
    # H′ chứng minh dư địa theo cụm là thật, nhưng nó chọn α mỗi cụm bằng NHÃN —
    # không triển khai được. Ở đây thử suy ra α từ thống kê của cụm, tính được
    # hoàn toàn từ đặc trưng.
    #
    # Cơ sở: §3.6c đo corr(chất lượng nhánh prompted, α*) = +0.42 ở mức lớp. Tín
    # hiệu theo TỪNG query chỉ với tới 3% dư địa vì quá nhiễu, nhưng trung bình
    # trên ~300 query mỗi cụm giảm nhiễu khoảng 17 lần.
    if best_cluster[0] > 0:
        k = best_cluster[0]
        cent = kmeans(g0, k, seed=0)
        assign = (q0 @ cent.t()).argmax(1)

        with torch.no_grad():
            agree = (uq * fq).sum(-1)
            topP = (mix(uq, fq, 1.0) @ mix(ug, fg, 1.0).t()).topk(10, -1).values
            topF = (mix(uq, fq, 0.0) @ mix(ug, fg, 0.0).t()).topk(10, -1).values
        qsig = {
            'agree': agree,
            'margin_prompted': topP[:, 0] - topP[:, -1],
            'top1_prompted': topP[:, 0],
            'margin_diff': (topP[:, 0] - topP[:, -1]) - (topF[:, 0] - topF[:, -1]),
        }
        del topP, topF

        cl_mask = [assign == j for j in range(k)]
        cl_mask = [m for m in cl_mask if m.any()]
        astar = np.array([max(args.alphas, key=lambda a: ap_by_alpha[a][m].mean().item())
                          for m in cl_mask])

        print(f'\n### H″ — α theo cụm KHÔNG dùng nhãn (k={k}, {len(cl_mask)} cụm)')
        print('  Tương quan giữa thống kê cụm (tính được từ đặc trưng) và α* của cụm:')
        best_name, best_r = None, 0.0
        cl_sig = {}
        for nm, s in qsig.items():
            v = np.array([s[m].mean().item() for m in cl_mask])
            cl_sig[nm] = v
            r = float(np.corrcoef(v, astar)[0, 1]) if v.std() > 0 else 0.0
            print(f'    {nm:<18} r = {r:+.3f}')
            if abs(r) > abs(best_r):
                best_name, best_r = nm, r

        print(f'  Tín hiệu mạnh nhất: {best_name} (r={best_r:+.3f})')

        # Quy tắc đơn điệu: xếp hạng cụm theo tín hiệu -> α trải đều trong [lo, hi].
        # Chỉ HAI tham số toàn cục, và chúng chọn trên nửa A rồi chấm trên nửa B —
        # yếu hơn hẳn so với chọn α riêng cho từng cụm bằng nhãn.
        v = cl_sig[best_name]
        rank = np.argsort(np.argsort(v if best_r > 0 else -v)) / max(len(v) - 1, 1)
        lo_hi = [(lo, hi) for lo in (0.2, 0.3, 0.4, 0.5) for hi in (0.5, 0.6, 0.7, 0.85, 1.0)
                 if hi > lo]
        B = ~half
        base_B = base_ap[B].mean().item()

        def rule_score(lo, hi, on):
            tot = 0.0
            for i, m in enumerate(cl_mask):
                a = min(args.alphas, key=lambda x: abs(x - (lo + (hi - lo) * rank[i])))
                mm = m & on
                if mm.any():
                    tot += ap_by_alpha[a][mm].sum().item()
            return tot / max(int(on.sum()), 1)

        lo_b, hi_b = max(lo_hi, key=lambda t: rule_score(*t, half))
        held_rule = rule_score(lo_b, hi_b, B)
        fixed_rule = rule_score(0.3, 0.7, B)      # dải cố định, KHÔNG fit gì
        print(f'\n  {"quy tắc":<34} {"mAP (nửa B)":>12} {"so với nền":>11}')
        print(f'  {"nền α=0.5":<34} {100 * base_B:>12.3f} {0.0:>+11.3f}')
        print(f'  {"α∈[0.3,0.7] theo hạng — 0 tham số":<34} {100 * fixed_rule:>12.3f} '
              f'{100 * (fixed_rule - base_B):>+11.3f}')
        print(f'  {f"α∈[{lo_b},{hi_b}] — 2 tham số fit trên A":<34} {100 * held_rule:>12.3f} '
              f'{100 * (held_rule - base_B):>+11.3f}')
        print(f'  {"α theo cụm bằng NHÃN (H′)":<34} {100 * best_cluster[1]:>12.3f} '
              f'{100 * (best_cluster[1] - base_B):>+11.3f}   <- trần của H″')
        for nm, v_ in (('H″ rule 0 tham số', fixed_rule), (f'H″ rule [{lo_b},{hi_b}]', held_rule)):
            rows.append({'method': nm, 'param': f'k={k},sig={best_name}',
                         'mAP': v_, 'delta': v_ - base_B})

    out = args.out or os.path.join(args.run_dir, 'improve_eval.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'CSV: {os.path.abspath(out)}')

    print(f"""
{'=' * 62}
Cách đọc:
  H′ — chỉ tin cột "giữ lại". Cột oracle luôn cao vì chọn α trên chính dữ
       liệu chấm điểm. Giữ lại > +0.31 pp (δ_min) mới là dư địa thật.
  F  — không cần train lại nên mọi mức dương đều dùng được ngay, nhưng phải
       báo cáo thành DÒNG RIÊNG trong bảng: nó dùng toàn bộ gallery lúc test
       (transductive), không so trực tiếp được với phương pháp inductive.
{'=' * 62}""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
