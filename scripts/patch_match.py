"""P1 — khớp ở mức PATCH (late interaction) thay vì chỉ CLS token.

Cơ sở (docs §8e). Dư địa còn lại tập trung ở 7 lớp yếu nhất: chiếm 34% query
với mAP 64.0, so với 89.4 của nhóm mạnh — chênh 25.4 pp. Kéo nhóm này lên 80 sẽ
cho mAP tổng 84.0 (+5.5 pp), lớn hơn αQE.

Ba lớp yếu nhất là `door` (50.1), `saw` (51.6), `window` (73.4) — đều là vật thể
nhỏ hoặc mảnh nằm trong ảnh CẢNH, trong khi sketch là vật thể cô lập chiếm hết
khung. CLS token của ảnh bị nền trung bình hoá. Nhưng ViT đã tính sẵn 50 patch
token mỗi ảnh rồi VỨT 49 cái.

    sim(sketch, photo) = mean_i max_j ⟨ patch_i^sketch , patch_j^photo ⟩

Mỗi patch của sketch tìm patch ảnh khớp nhất, không quan tâm phần còn lại của
cảnh. Đây là phương pháp INDUCTIVE — khác αQE (transductive, cần gallery), nên
nó đứng được thành đóng góp phương pháp chứ không chỉ một dòng hậu xử lý.

Chi phí: đầy đủ 52 token thì tốn ~50x so với CLS. `--tokens_k` giữ lại k token
có norm lớn nhất mỗi ảnh (norm cao = mang nhiều tín hiệu hơn), giảm chi phí
(k/52)² lần. k=8 chạy trong vài phút và đủ để biết hướng này có sống không.

Chạy:
    python scripts/patch_match.py --run_dir tb_logs/p4_alphasweep_s1/2026...
    python scripts/patch_match.py --run_dir ... --tokens_k 4 8 16 --branch mixed
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None

from scripts.sweep_alpha import build, find_ckpt, load_cfg, load_ckpt, mix  # noqa: E402


@torch.no_grad() if torch else (lambda f: f)
def extract_tokens(lit, loader, modality, device, branch, alpha, k):
    """Trả (tokens [N,k,D] đã chuẩn hoá, labels). Giữ k token có norm lớn nhất."""
    lit.eval()
    T, L = [], []
    for i, batch in enumerate(loader):
        tok = lit.extract_eval_tokens(batch[0].to(device), modality, branch=branch, alpha=alpha)
        if k > 0 and tok.shape[1] > k:
            idx = tok.norm(dim=-1).topk(k, dim=-1).indices          # [B, k]
            tok = torch.gather(tok, 1, idx.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        tok = tok / tok.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        T.append(tok.half().cpu()); L.append(batch[1].cpu())
        print(f'\r  {modality}: {i + 1}/{len(loader)} batch', end='', flush=True)
    print()
    return torch.cat(T), torch.cat(L)


def late_interaction(q_tok, g_tok, device, chunk=64):
    """sim[i,j] = mean_a max_b ⟨q[i,a], g[j,b]⟩ — chia lô theo query cho vừa VRAM."""
    Nq, Ng = len(q_tok), len(g_tok)
    out = torch.zeros(Nq, Ng, dtype=torch.float32)
    g = g_tok.to(device)                                    # [Ng, kg, D]
    gf = g.reshape(-1, g.shape[-1]).t().contiguous()        # [D, Ng*kg]
    kg = g.shape[1]
    for s in range(0, Nq, chunk):
        e = min(s + chunk, Nq)
        q = q_tok[s:e].to(device)                           # [c, kq, D]
        c, kq, _ = q.shape
        sim = (q.reshape(-1, q.shape[-1]) @ gf)             # [c*kq, Ng*kg]
        sim = sim.view(c, kq, Ng, kg).amax(-1).mean(1)      # max theo patch gallery, mean theo patch query
        out[s:e] = sim.float().cpu()
        del q, sim
        print(f'\r  late interaction: {e}/{Nq}', end='', flush=True)
    print()
    del g, gf
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--tokens_k', nargs='+', type=int, default=[4, 8, 16])
    ap.add_argument('--branch', default='mixed', choices=['mixed', 'prompted', 'frozen'])
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--device', default='cuda' if (torch and torch.cuda.is_available()) else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if torch is None:
        print('Cần cài torch.', file=sys.stderr)
        return 1

    from src.dataset_retrieval import ValidDataset
    from src.model_hicropl import retrieval_metrics, retrieval_topk

    cfg = load_cfg(args.run_dir, {'data_dir': args.data_dir,
                                  'test_batch_size': None, 'workers': None})
    ckpt = find_ckpt(cfg, args.ckpt)
    print(f'ckpt: {ckpt}\ndataset: {cfg.dataset} | device: {args.device} | branch: {args.branch}\n')

    lit = build(cfg, args.device)
    load_ckpt(lit, ckpt)
    mk = lambda m: DataLoader(ValidDataset(cfg, mode=m), batch_size=cfg.test_batch_size,
                              num_workers=cfg.workers, shuffle=False)

    # Mốc CLS để so — dùng đúng đường eval chuẩn
    from scripts.sweep_alpha import extract as extract_cls
    print('Mốc CLS:')
    uq, fq, lq = extract_cls(lit, mk('sketch'), 'sketch', args.device)
    ug, fg, lg = extract_cls(lit, mk('photo'), 'photo', args.device)
    dev = args.device
    q0 = mix(uq.to(dev), fq.to(dev), args.alpha)
    g0 = mix(ug.to(dev), fg.to(dev), args.alpha)
    lq, lg = lq.to(dev), lg.to(dev)
    base, _, _, _, map_k, p_k = retrieval_metrics(q0, g0, lq, lg, cfg.dataset)
    base = float(base)
    print(f'  CLS (nền): mAP@{map_k or "all"} = {100 * base:.3f}\n')
    del uq, fq, ug, fg, q0, g0
    torch.cuda.empty_cache() if dev == 'cuda' else None

    rows = [{'method': 'CLS (nền)', 'tokens_k': 1, 'mAP': base, 'delta': 0.0}]
    for k in args.tokens_k:
        print(f'--- tokens_k = {k} ---')
        qt, lq2 = extract_tokens(lit, mk('sketch'), 'sketch', dev, args.branch, args.alpha, k)
        gt, lg2 = extract_tokens(lit, mk('photo'), 'photo', dev, args.branch, args.alpha, k)
        sim = late_interaction(qt, gt, dev, args.chunk).to(dev)

        # retrieval_metrics nhận đặc trưng, không nhận ma trận sim. Ở đây đã có
        # sim trực tiếp nên tính metric bằng cùng công thức, dùng lại torchmetrics.
        from torchmetrics.functional.retrieval import (retrieval_average_precision,
                                                       retrieval_precision)
        n = len(sim)
        aps = torch.zeros(n, device=dev)
        prs = torch.zeros(n, device=dev)
        for i in range(n):
            tgt = (lg2.to(dev) == lq2[i].to(dev))
            aps[i] = (retrieval_average_precision(sim[i], tgt, top_k=min(map_k, len(gt)))
                      if map_k else retrieval_average_precision(sim[i], tgt))
            prs[i] = retrieval_precision(sim[i], tgt, top_k=p_k)
        v = float(aps.mean())
        print(f'  mAP@{map_k or "all"} = {100 * v:.3f}   ({100 * (v - base):+.3f} pp so với CLS)'
              f'   P@{p_k} = {100 * float(prs.mean()):.3f}')
        rows.append({'method': f'late interaction ({args.branch})', 'tokens_k': k,
                     'mAP': v, 'delta': v - base})
        del qt, gt, sim, aps, prs
        torch.cuda.empty_cache() if dev == 'cuda' else None

    out = args.out or os.path.join(args.run_dir, f'patch_match_{args.branch}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nCSV: {os.path.abspath(out)}')

    best = max(rows, key=lambda r: r['mAP'])
    print(f'\n{"=" * 62}')
    print(f'Tốt nhất: {best["method"]} k={best["tokens_k"]} -> {100 * best["mAP"]:.3f} '
          f'({100 * best["delta"]:+.3f} pp)')
    print(f"""
Cách đọc (δ_min ≈ 0.31 pp):
  Dương và tăng theo k  -> hướng sống; nâng k rồi chạy per-class để xem có đúng
                            nhóm door/saw/window được cải thiện không.
  Âm ở mọi k            -> CLS đã đủ; giả thuyết "mất thông tin patch" bị bác bỏ.
  Dương nhưng phẳng     -> chỉ vài patch mang tín hiệu, k nhỏ là đủ.

Khác αQE ở chỗ: đây là INDUCTIVE, không dùng gallery để sửa truy vấn, nên so
trực tiếp được với các phương pháp khác trong bảng.
{"=" * 62}""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
