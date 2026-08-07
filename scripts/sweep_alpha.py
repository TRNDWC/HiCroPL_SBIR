"""Quét tỉ lệ trộn residual α tại thời điểm EVAL, trên một checkpoint đã train.

    feat = norm( α · prompted + (1−α) · frozen )

Vì sao đáng làm: Giai đoạn 3a chỉ đo được α ≥ 0.5 (α học được luôn tăng về 1).
Nửa còn lại của đường cong hoàn toàn chưa biết, và **α = 0 chính là frozen-only**
— con số còn thiếu từ Giai đoạn 0. Một lần chạy trả lời cả hai.

Vì sao rẻ: hai nhánh prompted và frozen KHÔNG phụ thuộc α, chỉ phép trộn phụ
thuộc. Nên encoder chỉ chạy MỘT lần cho toàn bộ tập val, rồi mỗi α chỉ tốn một
phép trộn + một lần tính metric. Quét 9 giá trị α gần như bằng chi phí 1 lần eval.

Tự kiểm tra: α = 0.5 phải tái lập đúng mAP mà checkpoint đã báo lúc train. Script
in cảnh báo nếu lệch — đó là dấu hiệu nạp checkpoint sai hoặc dữ liệu khác.

Chạy:
    python scripts/sweep_alpha.py --run_dir tb_logs/p4_sweep/20260807-xxxxxx
    python scripts/sweep_alpha.py --run_dir ... --alphas 0 0.25 0.5 0.75 1.0
    python scripts/sweep_alpha.py --run_dir ... --ckpt path/to/x.ckpt --data_dir ../data/Sketchy
"""

import argparse
import csv
import glob
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # cho phép --help chạy khi chưa cài torch
    torch = None
    DataLoader = None


def load_cfg(run_dir, overrides):
    """Đọc config.json mà RunCSVLogger đã chụp lại — đảm bảo dựng đúng model."""
    path = os.path.join(run_dir, 'config.json')
    if not os.path.exists(path):
        raise SystemExit(f'Không thấy {path}. Cần run_dir của một lần chạy đã hoàn thành.')
    with open(path, encoding='utf-8') as f:
        cfg = SimpleNamespace(**json.load(f))
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    # Quét α thủ công nên phải tắt hai đường tắt trong extract_eval_features
    cfg.eval_frozen_only = False
    cfg.learn_mix_alpha = False
    return cfg


def find_ckpt(cfg, explicit):
    if explicit:
        return explicit
    d = os.path.join(getattr(cfg, 'save_dir', 'saved_models'), cfg.exp_name)
    hits = sorted(glob.glob(os.path.join(d, '*.ckpt')))
    hits = [h for h in hits if not h.endswith('last.ckpt')] or hits
    if not hits:
        raise SystemExit(
            f'Không thấy checkpoint trong {d}.\n'
            'Giai đoạn 3 chạy với --save_top_k=0 nên KHÔNG lưu checkpoint. '
            'Cần train lại một run với --save_top_k=1 (xem scripts/run_alpha_sweep.sh).')
    if len(hits) > 1:
        print(f'  (thấy {len(hits)} checkpoint, dùng {os.path.basename(hits[-1])})')
    return hits[-1]


def build(cfg, device):
    from src.model_hicropl import CustomCLIP, HiCroPL_SBIR
    from src.utils import load_clip_to_cpu, load_clip_to_cpu_teacher
    from src.dataset_retrieval import Sketchy

    train_ds = Sketchy(cfg, Sketchy.data_transform(cfg), mode='train', return_orig=False)
    classnames = list(train_ds.all_categories)

    clip_model = load_clip_to_cpu(cfg).to(device).float()
    clip_frozen = load_clip_to_cpu_teacher(cfg).to(device).float()
    custom = CustomCLIP(cfg, clip_model, clip_frozen, classnames=classnames)
    del clip_model, clip_frozen

    lit = HiCroPL_SBIR(cfg=cfg, args=cfg, classnames=classnames, model=custom).to(device)
    lit.print = print
    return lit


def load_ckpt(lit, path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    missing, unexpected = lit.load_state_dict(sd, strict=False)
    # mix_alpha_* chỉ tồn tại khi train với --learn_mix_alpha; thiếu là bình thường
    missing = [m for m in missing if 'mix_alpha' not in m]
    if missing:
        print(f'  ⚠ {len(missing)} key thiếu khi nạp checkpoint, vd: {missing[:5]}')
    if unexpected:
        print(f'  ⚠ {len(unexpected)} key thừa, vd: {unexpected[:5]}')
    print(f'  epoch trong checkpoint: {ck.get("epoch", "?")}')
    return ck


def extract(lit, loader, modality, device):
    """Chạy encoder MỘT lần, trả (prompted, frozen, labels) chưa trộn."""
    lit.eval()
    P, F, L = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            tensor, label = batch[0], batch[1]
            u, f = lit.extract_eval_branches(tensor.to(device), modality=modality)
            P.append(u.float().cpu()); F.append(f.float().cpu()); L.append(label.cpu())
            print(f'\r  {modality}: {i + 1}/{len(loader)} batch', end='', flush=True)
    print()
    return torch.cat(P), torch.cat(F), torch.cat(L)


def mix(u, f, a):
    m = a * u + (1.0 - a) * f
    return m / m.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', required=True, help='thư mục run chứa config.json')
    ap.add_argument('--ckpt', default=None, help='mặc định tự tìm trong save_dir/exp_name')
    ap.add_argument('--alphas', nargs='+', type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0])
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--test_batch_size', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--device',
                    default='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu')
    ap.add_argument('--out', default=None, help='CSV kết quả, mặc định <run_dir>/alpha_sweep.csv')
    ap.add_argument('--save_ap', action='store_true',
                    help='lưu vector AP của từng α để kiểm cặp bằng paired_test.py')
    args = ap.parse_args()

    if torch is None:
        print('Cần cài torch để chạy (chỉ --help là không cần).', file=sys.stderr)
        return 1

    cfg = load_cfg(args.run_dir, {'data_dir': args.data_dir,
                                  'test_batch_size': args.test_batch_size,
                                  'workers': args.workers})
    ckpt_path = find_ckpt(cfg, args.ckpt)
    print(f'config : {args.run_dir}/config.json')
    print(f'ckpt   : {ckpt_path}')
    print(f'dataset: {cfg.dataset} | device: {args.device}')

    from src.dataset_retrieval import ValidDataset
    from src.model_hicropl import retrieval_metrics, retrieval_topk

    lit = build(cfg, args.device)
    ck = load_ckpt(lit, ckpt_path)

    mk = lambda mode: DataLoader(ValidDataset(cfg, mode=mode),
                                 batch_size=cfg.test_batch_size,
                                 num_workers=cfg.workers, shuffle=False)
    print('\nTrích đặc trưng (encoder chỉ chạy một lần cho cả dải α):')
    uq, fq, lq = extract(lit, mk('sketch'), 'sketch', args.device)
    ug, fg, lg = extract(lit, mk('photo'), 'photo', args.device)
    print(f'  query {len(uq):,} sketch | gallery {len(ug):,} photo')

    dev = args.device
    uq, fq, ug, fg = uq.to(dev), fq.to(dev), ug.to(dev), fg.to(dev)
    lq, lg = lq.to(dev), lg.to(dev)

    map_k, p_k = retrieval_topk(cfg.dataset)
    rows, ap_store = [], {}
    print(f'\n{"α":>6} {"mAP@" + str(map_k or "all"):>10} {"P@" + str(p_k):>9}   ghi chú')
    print('-' * 52)
    for a in args.alphas:
        q, g = mix(uq, fq, a), mix(ug, fg, a)
        mAP, mP, apv, prv, _, _ = retrieval_metrics(q, g, lq, lg, cfg.dataset)
        note = {0.0: 'frozen-only (CLIP thuần)', 1.0: 'prompted-only'}.get(a, '')
        if abs(a - 0.5) < 1e-9:
            note = 'hành vi mặc định'
        print(f'{a:>6.2f} {100 * mAP.item():>10.3f} {100 * mP.item():>9.3f}   {note}')
        rows.append({'alpha': a, 'mAP': round(mAP.item(), 6),
                     f'P@{p_k}': round(mP.item(), 6), 'ckpt': os.path.basename(ckpt_path)})
        ap_store[a] = (apv.cpu().numpy(), prv.cpu().numpy())

    out = args.out or os.path.join(args.run_dir, 'alpha_sweep.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nCSV: {os.path.abspath(out)}')

    if args.save_ap:
        for a, (apv, prv) in ap_store.items():
            np.savez(os.path.join(args.run_dir, f'ap_alpha{a:.2f}.npz'),
                     ap=apv, precision=prv, sketch_labels=lq.cpu().numpy(),
                     epoch=np.array(ck.get('epoch', -1)), best_map=np.array(apv.mean()),
                     map_k=np.array(map_k), p_k=np.array(p_k))
        print(f'Vector AP: {args.run_dir}/ap_alpha*.npz — dùng được với paired_test.py')

    # -- diễn giải --
    best = max(rows, key=lambda r: r['mAP'])
    at05 = next((r for r in rows if abs(r['alpha'] - 0.5) < 1e-9), None)
    at00 = next((r for r in rows if r['alpha'] == 0.0), None)

    print(f'\n{"=" * 52}')
    print(f'Đỉnh tại α = {best["alpha"]:.2f}  ->  mAP {100 * best["mAP"]:.3f}')
    if at05:
        d = 100 * (best['mAP'] - at05['mAP'])
        print(f'So với α = 0.5 (mặc định): {d:+.3f} pp')
        print('  ⓘ Tự kiểm tra: mAP ở α=0.5 phải khớp con số checkpoint báo lúc train.')
        print('    Lệch nhiều -> nạp sai checkpoint hoặc khác dữ liệu, ĐỪNG tin bảng trên.')
    if at00:
        d = 100 * (at05['mAP'] - at00['mAP']) if at05 else None
        print(f'\nfrozen-only (α=0): mAP {100 * at00["mAP"]:.3f}'
              + (f'  -> cơ chế prompt đóng góp {d:+.3f} pp' if d is not None else ''))

    print(f'\nCách đọc (δ_min ≈ 0.31 pp từ Giai đoạn 0):')
    print('  đỉnh ở α < 0.5   -> nhánh prompted đang gây hại; đặt α tối ưu là cải thiện MIỄN PHÍ')
    print('  đỉnh ở α ≈ 0.5   -> 0.5 tối ưu thật, không phải trùng hợp; ưu tiên hướng B′')
    print('  đường cong phẳng -> α không phải đòn bẩy; chuyển sang hướng D/E/F')
    print('  đỉnh ở α > 0.5   -> giả định nền của hướng B được khôi phục')
    print('  Chênh lệch < δ_min thì dùng scripts/paired_test.py với --save_ap để kết luận chặt.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
