"""Hubness trong ZS-SBIR: chẩn đoán, rồi sửa bằng kỹ thuật mượn từ NLP.

BỐI CẢNH. Mọi hướng tăng năng lực thích nghi đều âm (11 hướng, ba đường liều–đáp
ứng đơn điệu — docs §6). Hướng duy nhất dương là αQE (+3.4 pp), tác động lên cấu
trúc gallery chứ không lên mô hình. Script này đi tiếp trên đúng trục đó, bằng
một họ kỹ thuật khác hẳn.

GIẢ THUYẾT. Truy hồi lân cận gần trong không gian nhiều chiều mắc bệnh *hubness*:
một số ít vector gallery trở thành láng giềng gần nhất của rất nhiều truy vấn,
bất kể nội dung. Bệnh nặng thêm khi hai không gian đến từ hai miền khác nhau và
ánh xạ giữa chúng không hoàn hảo — đúng tình huống ZS-SBIR với lớp chưa thấy.

Nếu đúng, nó giải thích các lớp yếu: `door` (50.1), `saw` (51.6) không nhất thiết
vì đặc trưng của chúng tồi, mà vì truy vấn của chúng bị hút vào hub của lớp khác.
Đây là cơ chế KHÁC hẳn "nền gây nhiễu" (PuXIM) và kiểm được riêng.

MƯỢN TỪ ĐÂU. Hubness được nghiên cứu kỹ ở dịch từ vựng song ngữ không song song
(Conneau et al., ICLR 2018) — cũng là hai không gian riêng, ánh xạ tuyến tính,
truy hồi chéo. Công cụ chính là CSLS. Ở truy hồi text–video có QB-Norm và chuẩn
hoá Sinkhorn. Chưa thấy ai mang sang ZS-SBIR.

QUAN SÁT LÀM GỌN CẢ HỌ PHƯƠNG PHÁP. Xếp hạng trong một truy vấn bất biến với mọi
số hạng chỉ phụ thuộc HÀNG. Nên trong truy hồi:

  · CSLS  csls(i,j) = 2·s(i,j) − r_Q(i) − r_G(j)  rút gọn thành  s(i,j) − β·r_G(j)
    — số hạng r_Q(i) biến mất hoàn toàn. Bilingual lexicon induction cần nó vì ở
    đó bài toán là ghép cặp toàn cục; truy hồi thì không.
  · Sinkhorn  s/τ + u(i) + v(j)  rút gọn thành  s(i,j) + τ·v(j).
  · Inverted softmax cũng chỉ là một hàm của cột.

Cả ba đều là **một độ lệch trên mỗi mục gallery**. Chúng chỉ khác nhau ở cách ước
lượng độ lệch đó. Điều này vừa làm code rẻ đi (không bao giờ phải hiện thực hoá
ma trận [Nq,Ng] đã sửa) vừa là phát biểu đáng viết: trong truy hồi, toàn bộ họ
"sửa hubness" thu về một chiều tự do trên mỗi mục gallery.

BIẾN THỂ RIÊNG CHO KIẾN TRÚC HAI NHÁNH. Ước lượng mật độ hub ở nhánh nào? Nhánh
prompted đã overfit (docs §6) nên cấu trúc lân cận của nó bị méo. Nhánh frozen
yếu hơn nhiều (30.0 so với 75.9) nhưng KHÔNG bị thích nghi, nên mật độ của nó
phản ánh ngữ nghĩa chung. `--hub_branch frozen` tính r_G ở nhánh frozen rồi áp
cho độ tương đồng của nhánh mixed. Phần này không mượn của ai.

Chạy:
    python scripts/hubness.py --run_dir tb_logs/p4_alphasweep_s1/2026...
    python scripts/hubness.py --run_dir ... --hub_branch frozen --save_ap
"""

import argparse
import csv
import json
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

from scripts.improve_eval import alpha_qe, l2  # noqa: E402
from scripts.sweep_alpha import build, extract, find_ckpt, load_cfg, load_ckpt, mix  # noqa: E402

CHUNK = 1024


# --------------------------------------------------------------------------
# Chấm điểm. retrieval_metrics chỉ nhận đặc trưng, mà mọi phép sửa ở đây tác
# động lên CỘT của ma trận sim, nên cần đường tính riêng — có tự kiểm đối chiếu.
#
# Với sketchy, map_k = 0 nghĩa là mAP@all: K = Ng = 12552. Một topk như thế trên
# [12694, 12552] cấp phát 1.27 GB chỉ riêng chỉ số int64. Nên chia lô theo hàng
# và KHÔNG BAO GIỜ hiện thực hoá ma trận sim đã sửa — độ lệch cột cộng vào từng lô.
# --------------------------------------------------------------------------

def score(qf, gf, lq, lg, map_k, p_k, col_off=None, chunk=CHUNK):
    """Trả (vector AP, vector P@k) cho sim = qf @ gf.T + col_off.

    Khớp định nghĩa torchmetrics: trong top-K, AP = trung bình của
    (hạng-trong-nhóm-đúng / vị trí) trên các mục đúng; mẫu số là số mục đúng
    TRONG TOP-K, không phải tổng số mục đúng.
    """
    Nq, Ng = len(qf), len(gf)
    K = min(map_k or Ng, Ng)
    pos = torch.arange(1, K + 1, device=qf.device, dtype=torch.float32)
    APs, PKs = [], []
    for s in range(0, Nq, chunk):
        sim = qf[s:s + chunk] @ gf.t()
        if col_off is not None:
            sim = sim + col_off[None, :]
        idx = sim.topk(K, dim=-1).indices
        del sim
        relf = (lg[idx] == lq[s:s + chunk, None]).float()
        del idx
        nrel = relf.sum(1)
        ap = torch.where(nrel > 0, ((relf.cumsum(1) / pos) * relf).sum(1) / nrel.clamp(min=1),
                         torch.zeros_like(nrel))
        APs.append(ap)
        PKs.append(relf[:, :min(p_k, K)].mean(1))
        del relf
    return torch.cat(APs), torch.cat(PKs)


def tie_stats(qf, gf, K, chunk=CHUNK, rows=None):
    """Đếm hoà điểm trong top-K: (tỉ lệ truy vấn có hoà, số cặp hoà trung bình).

    Quan trọng gấp đôi ở đây. Thứ nhất, hoà điểm giải thích vì sao hai cách cài
    AP đúng như nhau vẫn lệch: `topk` phá hoà theo thứ tự chỉ số, và kernel 1-D
    (torchmetrics gọi trên từng hàng) khác kernel 2-D (gọi trên cả lô).

    Thứ hai — và đây mới là chỗ nguy hiểm — MỌI phép sửa trong script này là một
    độ lệch trên mỗi cột. Cộng một độ lệch vào các mục đang hoà sẽ PHÁ HOÀ, nên
    tạo ra thay đổi mAP không đến từ cơ chế hubness nào cả. Tỉ lệ hoà là cận
    trên của phần "ăn may" đó, nên phải đo và báo cáo.
    """
    q = qf if rows is None else qf[rows]
    tot = nq = 0
    pairs = 0.0
    for s in range(0, len(q), chunk):
        v = (q[s:s + chunk] @ gf.t()).topk(K, dim=-1).values
        eq = v[:, 1:] == v[:, :-1]
        tot += int(eq.any(1).sum())
        pairs += float(eq.sum())
        nq += len(v)
    return tot / max(nq, 1), pairs / max(nq, 1)


def hub_scores(sim, k):
    """r_G(j) = trung bình top-k độ tương đồng của gallery j sang phía TRUY VẤN."""
    return sim.topk(k, dim=0).values.mean(0)              # [Ng]


def standardize(rg, unit):
    """Đưa r_G về thang đo chung để β so sánh được GIỮA CÁC NHÁNH.

    Nhánh frozen và prompted có phân bố độ tương đồng khác hẳn nhau (mAP 30.0 so
    với 75.9), nên r_G thô của chúng lệch nhau cả về tâm lẫn độ trải. Trừ thẳng
    `β·r_G` thì cùng một β lại là hai cường độ khác nhau, và bảng kết quả sẽ so
    thang đo chứ không so cơ chế.

    Trừ trung bình là hằng số trên mọi cột nên không đổi thứ hạng — chỉ phép chia
    độ lệch chuẩn mới thực sự có tác dụng. Sau chuẩn hoá, β tính bằng đơn vị "độ
    lệch chuẩn của sim nền", nên cùng nghĩa ở mọi nhánh.
    """
    return (rg - rg.mean()) / rg.std().clamp(min=1e-8) * unit


def sinkhorn_offset(sim, tau, iters, chunk=CHUNK):
    """Độ lệch cột τ·v của chuẩn hoá Sinkhorn trên exp(s/τ).

    Cân cả xác suất truy vấn lẫn xác suất gallery, nên ép mỗi mục gallery nhận
    lượng khối lượng tương đương — dạng mạnh hơn CSLS, vốn chỉ trừ một hằng số
    ước lượng từ top-k. Chỉ trả v: u(i) là hằng số trên mỗi hàng nên vô hiệu.
    """
    Nq, Ng = sim.shape
    u = torch.zeros(Nq, device=sim.device)
    v = torch.zeros(Ng, device=sim.device)
    for _ in range(iters):
        for s in range(0, Nq, chunk):
            u[s:s + chunk] = -torch.logsumexp(sim[s:s + chunk] / tau + v[None, :], dim=1)
        # logsumexp theo CỘT phải gộp qua các lô: logsumexp từng lô rồi logsumexp chồng
        parts = [torch.logsumexp(sim[s:s + chunk] / tau + u[s:s + chunk, None], dim=0)
                 for s in range(0, Nq, chunk)]
        v = -torch.logsumexp(torch.stack(parts), dim=0)
    return tau * v


def all_but_top(q, g, d, fit='gallery'):
    """Bỏ d thành phần chính đầu tiên (Mu & Viswanath, ICLR 2018).

    Không gian CLIP dị hướng: vài hướng phương sai lớn chi phối tích vô hướng và
    chính chúng sinh hub. Đây là phép sửa duy nhất trong script tác động lên ĐẶC
    TRƯNG chứ không lên cột sim.
    """
    if d <= 0:
        return q, g
    base = g if fit == 'gallery' else torch.cat([q, g], 0)
    mu = base.mean(0, keepdim=True)
    c = base - mu
    cov = (c.t() @ c) / max(len(c) - 1, 1)                 # [D,D] rẻ hơn SVD trên [N,D]
    evec = torch.linalg.eigh(cov.double()).eigenvectors    # trị riêng TĂNG dần
    evec = evec.flip(-1)[:, :d].to(q.dtype)                # -> d hướng phương sai lớn nhất
    rm = lambda x: l2((x - mu) - ((x - mu) @ evec) @ evec.t())
    return rm(q), rm(g)


def n_occurrence(sim, k):
    """N_k(j) = số truy vấn có j trong top-k. Thước đo hubness kinh điển."""
    idx = sim.topk(k, dim=-1).indices
    return torch.bincount(idx.reshape(-1), minlength=sim.shape[1]), idx


def skew(x):
    x = x.float()
    return float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument('--run_dir', required=True)
    ap_.add_argument('--ckpt', default=None)
    ap_.add_argument('--alpha', type=float, default=0.5)
    ap_.add_argument('--hub_k', nargs='+', type=int, default=[5, 10, 20, 50])
    ap_.add_argument('--beta', nargs='+', type=float, default=[0.25, 0.5, 0.75, 1.0],
                     help='cường độ trừ hub; 0.5 = CSLS gốc')
    ap_.add_argument('--hub_branch', nargs='+', default=['mixed', 'frozen'],
                     choices=['mixed', 'frozen', 'prompted'],
                     help='nhánh ước lượng mật độ hub; nhiều giá trị chạy chung '
                          'MỘT lần trích đặc trưng')
    ap_.add_argument('--abt_d', nargs='+', type=int, default=[0, 1, 2, 4, 8])
    ap_.add_argument('--tau', nargs='+', type=float, default=[0.02, 0.05, 0.1])
    ap_.add_argument('--sinkhorn_iters', type=int, default=3)
    ap_.add_argument('--qe_k', type=int, default=75, help='αQE để kiểm cộng dồn; 0 = bỏ')
    ap_.add_argument('--qe_power', type=float, default=3.0)
    ap_.add_argument('--nk', type=int, default=10)
    ap_.add_argument('--chunk', type=int, default=CHUNK)
    ap_.add_argument('--data_dir', default=None)
    ap_.add_argument('--device', default='cuda' if (torch and torch.cuda.is_available()) else 'cpu')
    ap_.add_argument('--save_ap', action='store_true')
    ap_.add_argument('--out', default=None)
    args = ap_.parse_args()

    if torch is None:
        print('Cần cài torch.', file=sys.stderr)
        return 1

    from src.dataset_retrieval import ValidDataset
    from src.model_hicropl import retrieval_metrics, retrieval_topk

    cfg = load_cfg(args.run_dir, {'data_dir': args.data_dir,
                                  'test_batch_size': None, 'workers': None})
    ckpt = find_ckpt(cfg, args.ckpt)
    print(f'ckpt: {ckpt}\ndataset: {cfg.dataset} | device: {args.device}\n')

    lit = build(cfg, args.device)
    load_ckpt(lit, ckpt)
    mk = lambda m: DataLoader(ValidDataset(cfg, mode=m), batch_size=cfg.test_batch_size,
                              num_workers=cfg.workers, shuffle=False)
    print('Trích đặc trưng (một lần):')
    uq, fq, lq = extract(lit, mk('sketch'), 'sketch', args.device)
    ug, fg, lg = extract(lit, mk('photo'), 'photo', args.device)
    del lit
    torch.cuda.empty_cache() if args.device == 'cuda' else None

    dev = args.device
    uq, fq, ug, fg = uq.to(dev), fq.to(dev), ug.to(dev), fg.to(dev)
    lq, lg = lq.to(dev), lg.to(dev)
    map_k, p_k = retrieval_topk(cfg.dataset)
    ch = args.chunk

    q0, g0 = mix(uq, fq, args.alpha), mix(ug, fg, args.alpha)
    sim0 = q0 @ g0.t()

    # -- tự kiểm: đường tính AP mới phải khớp retrieval_metrics của repo --------
    ap0, _ = score(q0, g0, lq, lg, map_k, p_k, chunk=ch)
    _, _, ref_ap, _, _, _ = retrieval_metrics(q0, g0, lq, lg, cfg.dataset)
    dd = (ap0 - ref_ap).abs()
    dmax, dmean = float(dd.max()), float(dd.mean())
    nd = int((dd > 1e-6).sum())
    del ref_ap
    K = min(map_k or len(g0), len(g0))
    print(f'\nTự kiểm score() vs retrieval_metrics:')
    print(f'  lệch tối đa {dmax:.2e} | trung bình {dmean:.2e} | '
          f'{nd}/{len(dd)} truy vấn lệch')

    # Sai LOGIC thì lệch có hệ thống trên nhiều truy vấn. Phá hoà thì lệch nhỏ,
    # thưa, và chỉ ở những truy vấn có hoà điểm. Hai trường hợp phân biệt được.
    if dmean > 1e-6 or dmax > 5e-3:
        print('  ✗ Lệch có hệ thống — sai logic, KHÔNG phải phá hoà. Dừng.')
        return 1
    tie_all, pairs_all = tie_stats(q0, g0, K, ch)
    if nd:
        tie_bad, _ = tie_stats(q0, g0, K, ch, rows=(dd > 1e-6).nonzero(as_tuple=True)[0])
        print(f'  hoà điểm trong top-{K}: {100 * tie_all:.2f}% truy vấn nói chung, '
              f'{100 * tie_bad:.2f}% trong nhóm lệch')
        if tie_bad < 0.9:
            print('  ✗ Nhóm lệch KHÔNG phải do hoà điểm — còn nguyên nhân khác. Dừng.')
            return 1
        print('  OK — lệch chỉ đến từ phá hoà, không phải sai logic.')
    else:
        print('  OK — khớp tuyệt đối.')

    # Cận trên của phần "ăn may": mọi phép sửa ở phần B là một độ lệch trên mỗi
    # cột, nên nó PHÁ HOÀ và có thể đổi mAP mà không cần cơ chế hubness nào.
    print(f'\n⚠ Hoà điểm trong top-{K}: {100 * tie_all:.2f}% truy vấn, '
          f'trung bình {pairs_all:.2f} cặp/truy vấn.')
    print('  Mọi phép sửa ở phần B là độ lệch theo cột nên nó phá hoà. Con số trên')
    print('  là cận trên của mức thay đổi mAP KHÔNG đến từ cơ chế hubness.')

    base = float(ap0.mean())
    print(f'\nNỀN α={args.alpha}: mAP@{map_k or "all"} = {100 * base:.3f}')

    # ======================================================================
    # A. CHẨN ĐOÁN — hubness có thật không, và có trùng nhóm lớp yếu không
    # ======================================================================
    print('\n' + '=' * 70)
    print('A. CHẨN ĐOÁN HUBNESS')
    print('=' * 70)
    diag = {'baseline_mAP': base, 'nk': args.nk, 'alpha': args.alpha,
            'tie_frac': tie_all, 'tie_pairs': pairs_all,
            'selfcheck_dmax': dmax, 'selfcheck_dmean': dmean}
    print(f'\nĐộ lệch (skewness) của N_{args.nk} — càng cao càng nhiều hub:')
    print(f'{"không gian":<26}{"skew":>9}{"N_k max":>10}{"gallery không bao giờ vào top":>32}')
    print('-' * 77)
    for nm, key_, a in (('frozen (α=0)', 'frozen', 0.0), ('prompted (α=1)', 'prompted', 1.0),
                        (f'mixed (α={args.alpha})', 'mixed', args.alpha)):
        s = sim0 if a == args.alpha else mix(uq, fq, a) @ mix(ug, fg, a).t()
        nk_, _ = n_occurrence(s, args.nk)
        diag[f'skew_{key_}'] = skew(nk_)
        diag[f'orphan_{key_}'] = float((nk_ == 0).float().mean())
        print(f'{nm:<26}{skew(nk_):>9.3f}{int(nk_.max()):>10}'
              f'{100 * float((nk_ == 0).float().mean()):>31.1f}%')
        if s is not sim0:
            del s
        del nk_
    # Mốc cùng miền: tách phần hubness do NHIỀU CHIỀU khỏi phần do KHOẢNG CÁCH MIỀN.
    s_ss = g0 @ g0.t()
    s_ss.fill_diagonal_(-2)
    nk_ss, _ = n_occurrence(s_ss, args.nk)
    diag['skew_photo2photo'] = skew(nk_ss)
    diag['orphan_photo2photo'] = float((nk_ss == 0).float().mean())
    print(f'{"photo->photo (cùng miền)":<26}{skew(nk_ss):>9.3f}{int(nk_ss.max()):>10}'
          f'{100 * float((nk_ss == 0).float().mean()):>31.1f}%')
    del s_ss, nk_ss

    nk, idx = n_occurrence(sim0, args.nk)
    thr = torch.quantile(nk.float(), 0.99)
    is_hub = nk.float() >= thr
    rel = lg[idx] == lq[:, None]
    hub_hit = is_hub[idx]
    bad_hub = (~rel) & hub_hit
    print(f'\nHub = 1% gallery có N_k cao nhất (N_k >= {int(thr)}).')
    print(f'  Chiếm {100 * float(hub_hit.float().mean()):.1f}% tổng số ô top-{args.nk} '
          f'(phân bố đều thì phải 1.0%).')
    print(f'  Trong đó SAI lớp: {100 * float(bad_hub.float().mean()):.1f}% tổng số ô.')

    cls = torch.unique(lq).tolist()
    rows_c = sorted(((c, float(ap0[lq == c].mean()), float(bad_hub[lq == c].float().mean()),
                      int((lq == c).sum())) for c in cls), key=lambda r: r[1])
    mp = np.array([r[1] for r in rows_c])
    bh = np.array([r[2] for r in rows_c])
    r_corr = float(np.corrcoef(mp, bh)[0, 1]) if bh.std() > 0 else 0.0
    diag.update(r_class_mAP_vs_badhub=r_corr,
                hub_share=float(hub_hit.float().mean()),
                bad_hub_share=float(bad_hub.float().mean()),
                n_classes=len(rows_c))
    print(f'\nTương quan mAP của lớp vs tỉ lệ bị hút vào hub sai lớp: r = {r_corr:+.3f}')
    print('  (âm mạnh = lớp yếu đúng là lớp bị hub nuốt -> giả thuyết được ủng hộ)')
    print(f'\n{"lớp":>5}{"mAP":>9}{"% ô là hub sai lớp":>21}{"n truy vấn":>12}')
    print('-' * 47)
    for c, m_, b_, n_ in rows_c[:7]:
        print(f'{c:>5}{100 * m_:>9.2f}{100 * b_:>20.1f}%{n_:>12}')
    print(f'{"...":>5}')
    for c, m_, b_, n_ in rows_c[-3:]:
        print(f'{c:>5}{100 * m_:>9.2f}{100 * b_:>20.1f}%{n_:>12}')
    print(f'\nLớp sở hữu nhiều hub nhất: '
          f'{torch.bincount(lg[is_hub], minlength=int(lg.max()) + 1).topk(5).indices.tolist()}')
    del rel, hub_hit, idx, nk

    # ======================================================================
    # B. SỬA — mọi phép đều TRANSDUCTIVE (dùng thống kê gallery lúc test)
    # ======================================================================
    print('\n' + '=' * 70)
    print('B. SỬA')
    print('=' * 70)
    rows = [{'method': 'baseline', 'param': f'α={args.alpha}', 'mAP': base, 'delta': 0.0}]
    store = {'baseline': ap0.cpu().numpy()}

    def rec(name, param, apv):
        v = float(apv.mean())
        rows.append({'method': name, 'param': param, 'mAP': v, 'delta': v - base})
        return v

    # β tính bằng đơn vị độ lệch chuẩn của sim nền (xem standardize) để so được
    # giữa các nhánh. CSLS thô báo cáo riêng ở dưới để không mất phương pháp gốc.
    unit = sim0.std()
    # -- B0. ĐỐI CHỨNG GIẢ DƯỢC --------------------------------------------
    # Độ lệch cột NGẪU NHIÊN, cùng độ trải với r̂_G. Nó không mang thông tin
    # hubness nào, nên mọi thay đổi mAP mà nó tạo ra đều là phá hoà cộng với
    # nhiễu thuần. Mức tăng của CSLS chỉ có nghĩa khi vượt hẳn mức này.
    print('\n### B0. Đối chứng — độ lệch cột NGẪU NHIÊN cùng độ trải')
    g_ = torch.Generator(device='cpu').manual_seed(0)
    plac = []
    for i in range(5):
        off = torch.randn(len(g0), generator=g_).to(dev) * unit
        for b in args.beta:
            apv, _ = score(q0, g0, lq, lg, map_k, p_k, col_off=-b * off, chunk=ch)
            plac.append(float(apv.mean()) - base)
        del off
    plac_abs = max(abs(x) for x in plac)
    print(f'  {len(plac)} lần thử (5 hạt giống × {len(args.beta)} β): '
          f'thay đổi trong khoảng [{100 * min(plac):+.3f}, {100 * max(plac):+.3f}] pp')
    print(f'  -> NGƯỠNG GIẢ DƯỢC = {100 * plac_abs:.3f} pp. Mức tăng dưới ngưỡng này')
    print('     không phân biệt được với phá hoà.')
    rows.append({'method': 'giả dược (lệch cột ngẫu nhiên)',
                 'param': f'|max| trên {len(plac)} lần', 'mAP': base + plac_abs,
                 'delta': plac_abs})

    print(f'\n### B1+B3. Trừ hub và Sinkhorn, theo NHÁNH ước lượng mật độ')
    print(f'    β theo đơn vị σ(sim) = {float(unit):.4f}')
    print('    Mỗi nhánh dựng ma trận sim một lần rồi giải phóng — nhiều nhánh KHÔNG')
    print('    tốn thêm lần trích đặc trưng nào.')
    best_csls = (base, args.hub_k[0], args.beta[0], args.hub_branch[0])
    best_sk = (base, args.tau[0], args.hub_branch[0])
    for br in args.hub_branch:
        sim_hub = sim0 if br == 'mixed' else (
            lambda a: mix(uq, fq, a) @ mix(ug, fg, a).t())(1.0 if br == 'prompted' else 0.0)
        print(f'\n  -- nhánh ước lượng: {br} --')
        print(f'  {"k":>5}{"β":>7}{"mAP":>10}{"so với nền":>12}')
        print('  ' + '-' * 34)
        for k in args.hub_k:
            rg = standardize(hub_scores(sim_hub, k), unit)
            for b in args.beta:
                apv, _ = score(q0, g0, lq, lg, map_k, p_k, col_off=-b * rg, chunk=ch)
                v = rec('CSLS', f'k={k},β={b},branch={br}', apv)
                flag = ' <-' if v > best_csls[0] else ''
                print(f'  {k:>5}{b:>7.2f}{100 * v:>10.3f}{100 * (v - base):>+12.3f}{flag}')
                if v > best_csls[0]:
                    best_csls = (v, k, b, br)
                    store['csls'] = apv.cpu().numpy()
            del rg
        print(f'  {"τ (Sinkhorn)":>12}{"mAP":>10}{"so với nền":>12}')
        print('  ' + '-' * 34)
        for t in args.tau:
            off = sinkhorn_offset(sim_hub, t, args.sinkhorn_iters, chunk=ch)
            apv, _ = score(q0, g0, lq, lg, map_k, p_k, col_off=off, chunk=ch)
            v = rec('sinkhorn', f'τ={t},branch={br}', apv)
            flag = ' <-' if v > best_sk[0] else ''
            print(f'  {t:>12.3f}{100 * v:>10.3f}{100 * (v - base):>+12.3f}{flag}')
            if v > best_sk[0]:
                best_sk = (v, t, br)
                store['sinkhorn'] = apv.cpu().numpy()
            del off
        if sim_hub is not sim0:
            del sim_hub
        torch.cuda.empty_cache() if dev == 'cuda' else None

    # CSLS đúng như bài gốc: r_G thô trên nhánh mixed, β = 0.5. Phải có dòng này
    # để bảng báo cáo được phương pháp tham chiếu, không chỉ biến thể đã chỉnh.
    rg_raw = hub_scores(sim0, 10)
    apv, _ = score(q0, g0, lq, lg, map_k, p_k, col_off=-0.5 * rg_raw, chunk=ch)
    v_raw = rec('CSLS gốc (thô)', 'k=10,β=0.5,branch=mixed', apv)
    print(f'\n  CSLS gốc (r_G thô, nhánh mixed, k=10, β=0.5): {100 * v_raw:.3f} '
          f'({100 * (v_raw - base):+.3f} pp)')
    store['csls_raw'] = apv.cpu().numpy()
    del rg_raw

    print('\n### B2. all-but-the-top — bỏ d hướng phương sai lớn nhất')
    print(f'{"d":>5}{"mAP":>10}{"so với nền":>12}')
    print('-' * 27)
    best_abt = (base, 0)
    for d in args.abt_d:
        qw, gw = all_but_top(q0, g0, d)
        apv, _ = score(qw, gw, lq, lg, map_k, p_k, chunk=ch)
        v = rec('all-but-top', f'd={d}', apv)
        flag = ' <-' if v > best_abt[0] else ''
        print(f'{d:>5}{100 * v:>10.3f}{100 * (v - base):>+12.3f}{flag}')
        if v > best_abt[0]:
            best_abt = (v, d)
            store['abt'] = apv.cpu().numpy()
        del qw, gw

    # B4. cộng dồn với αQE — αQE sửa TRUY VẤN, trừ hub sửa CỘT gallery
    if args.qe_k > 0:
        print(f'\n### B4. Kết hợp với αQE (qe_k={args.qe_k})')
        qA = alpha_qe(q0, g0, sim0, args.qe_k, args.qe_power)
        ap_qe, _ = score(qA, g0, lq, lg, map_k, p_k, chunk=ch)
        v_qe = rec('αQE', f'k={args.qe_k}', ap_qe)
        store['qe'] = ap_qe.cpu().numpy()

        # dựng lại sim của nhánh thắng — nó đã được giải phóng sau vòng B1+B3
        _, kb, bb, brb = best_csls
        sh = sim0 if brb == 'mixed' else (
            lambda a: mix(uq, fq, a) @ mix(ug, fg, a).t())(1.0 if brb == 'prompted' else 0.0)
        rg = standardize(hub_scores(sh, kb), unit)
        if sh is not sim0:
            del sh
        ap_b, _ = score(qA, g0, lq, lg, map_k, p_k, col_off=-bb * rg, chunk=ch)
        v_both = rec('αQE + CSLS', f'k={args.qe_k},β={bb},branch={brb}', ap_b)
        store['qe_csls'] = ap_b.cpu().numpy()

        print(f'  nền           {100 * base:>9.3f}')
        print(f'  chỉ CSLS      {100 * best_csls[0]:>9.3f}  {100 * (best_csls[0] - base):+.3f}'
              f'   [k={kb}, β={bb}, nhánh={brb}]')
        print(f'  chỉ αQE       {100 * v_qe:>9.3f}  {100 * (v_qe - base):+.3f}')
        print(f'  CẢ HAI        {100 * v_both:>9.3f}  {100 * (v_both - base):+.3f}')
        add = (v_both - base) - (best_csls[0] - base) - (v_qe - base)
        print(f'  -> {"cộng dồn" if add > -0.002 else "giẫm chân nhau"} '
              f'(chênh so với tổng hai phần: {100 * add:+.3f} pp)')
        del qA, rg

    # -- tốt nhất + phân rã theo lớp (đây mới là chỗ xác nhận CƠ CHẾ) ------
    # Dòng giả dược là NGƯỠNG, không phải một phương pháp — loại khỏi phép chọn.
    best = max((r for r in rows if not r['method'].startswith('giả dược')),
               key=lambda r: r['mAP'])
    print('\n' + '=' * 70)
    print(f'Tốt nhất: {best["method"]} ({best["param"]}) -> {100 * best["mAP"]:.3f} '
          f'({100 * best["delta"]:+.3f} pp)')
    verdict = ('VƯỢT giả dược' if best['delta'] > plac_abs else
               'KHÔNG vượt giả dược — không phân biệt được với phá hoà')
    print(f'  ngưỡng giả dược {100 * plac_abs:.3f} pp  |  δ_min 0.31 pp  ->  {verdict}')
    key = {'CSLS': 'csls', 'CSLS gốc (thô)': 'csls_raw', 'all-but-top': 'abt',
           'sinkhorn': 'sinkhorn', 'αQE': 'qe', 'αQE + CSLS': 'qe_csls'}.get(best['method'])
    if key in store:
        d = torch.from_numpy(store[key]).to(dev) - ap0
        print('\nPhân rã theo lớp (7 lớp yếu nhất — nơi giả thuyết dự đoán tăng nhiều nhất):')
        print(f'{"lớp":>5}{"mAP nền":>10}{"thay đổi":>11}')
        print('-' * 26)
        weak = torch.zeros_like(lq, dtype=torch.bool)
        for c, m_, _, _ in rows_c[:7]:
            print(f'{c:>5}{100 * m_:>10.2f}{100 * float(d[lq == c].mean()):>+11.3f}')
            weak |= lq == c
        diag.update(best_method=best['method'], best_param=best['param'],
                    best_delta=best['delta'], placebo_threshold=plac_abs,
                    gain_weak=float(d[weak].mean()), gain_rest=float(d[~weak].mean()))
        print(f'\n  nhóm yếu {100 * float(d[weak].mean()):+.3f} pp   |   '
              f'nhóm còn lại {100 * float(d[~weak].mean()):+.3f} pp')
        print('  Giả thuyết đúng <=> nhóm yếu tăng NHIỀU HƠN HẲN nhóm còn lại.')

    out = args.out or os.path.join(args.run_dir, 'hubness.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'\nCSV : {os.path.abspath(out)}')
    # Số chẩn đoán phần A chỉ được IN ra, mà chúng mới là phần quyết định câu
    # chuyện — nên ghi ra JSON để tổng hợp qua nhiều seed.
    dj = os.path.splitext(out)[0] + '_diag.json'
    with open(dj, 'w', encoding='utf-8') as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f'Diag: {os.path.abspath(dj)}')

    if args.save_ap:
        for nm, v in store.items():
            np.savez(os.path.join(args.run_dir, f'ap_hub_{nm}.npz'), ap=v,
                     sketch_labels=lq.cpu().numpy(), best_map=np.array(v.mean()),
                     map_k=np.array(map_k), p_k=np.array(p_k), epoch=np.array(-1))
        print(f'Vector AP: {args.run_dir}/ap_hub_*.npz — dùng với paired_test.py')

    print(f"""
{'=' * 70}
Cách đọc (δ_min ≈ 0.31 pp):
  Phần A quyết định câu chuyện, phần B chỉ là con số.
    skew(mixed) >> skew(photo->photo) -> hubness đến từ KHOẢNG CÁCH MIỀN, không
        phải từ số chiều. Đây là phát biểu đáng viết.
    r âm mạnh                         -> lớp yếu đúng là lớp bị hub nuốt.
    nhóm yếu tăng nhiều hơn nhóm mạnh -> cơ chế được xác nhận, không phải một
        phép hậu xử lý ăn may.
  Nếu A âm mà B vẫn dương thì chỉ còn là thủ thuật, không phải đóng góp.
{'=' * 70}""")
    return 0


if __name__ == '__main__':
    sys.exit(main())
