"""Quét các bất biến của training loop HiCroPL-SBIR trên CLIP thật.

Script này KHÔNG thay thế unit test trong `tests/` - nó bổ sung phần mà mock
không bắt được. Toàn bộ leak parameters đã sửa đều bắt nguồn từ cấu trúc module
thật của CLIP (`nn.MultiheadAttention.in_proj_weight` là bare Parameter, không
phải `.weight`), nên phải dựng model thật mới phát hiện được.

Chạy:
    python scripts/verify_training.py                  # config mặc định
    python scripts/verify_training.py --all            # quét mọi cấu hình ablation
    python scripts/verify_training.py --device cuda    # ép chạy GPU
    python scripts/verify_training.py --list           # xem danh sách check

Exit code 0 nếu tất cả PASS, 1 nếu có FAIL. Dùng được trong CI.

Lưu ý tài nguyên: CustomCLIP giữ 4 bản CLIP ViT-B/32 (~2.4GB fp32). Lần chạy
đầu tải weights (~350MB) về cache của `src/clip`. Trên CPU, một config mất
khoảng 1-3 phút vì có 4 lượt forward+backward độc lập; trên GPU thì vài giây.
"""

import argparse
import os
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Console Windows mặc định cp1252, không encode được tiếng Việt trong output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

try:
    import torch
except ImportError:  # cho phép `--list` chạy trong môi trường chưa cài torch
    torch = None

# Ngưỡng cho check `trainable_budget`. LayerNorm của 2 student ~131K, prompt
# tokens ~55K, 4 mạng CrossPromptAttention + các AttentionPooling ~7M.
# Trước khi sửa freeze policy, con số này là ~126M.
MAX_TRAINABLE_PARAMS = 20_000_000

# Tên các bare nn.Parameter của CLIP mà `freeze_all_but_bn` không chạm tới được.
CLIP_BARE_PARAMS = ('text_projection', 'positional_embedding', 'proj',
                    'class_embedding', 'logit_scale', 'in_proj_weight', 'in_proj_bias')

LEARNER_PREFIXES = ('visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch')
DISTILL_PREFIXES = ('clip_distill_photo', 'clip_distill_sketch', 'clip_distill')


# --------------------------------------------------------------------------
# Dựng model
# --------------------------------------------------------------------------

def build_cfg(**overrides):
    """Config tối thiểu đủ cho CustomCLIP + HiCroPL_SBIR.

    Không import `experiments.options` vì file đó gọi `parse_args()` ngay lúc
    import, sẽ nuốt argv của script này.
    """
    cfg = SimpleNamespace(
        backbone='ViT-B/32',
        clip_trainer='HiCroPL',
        prompt_depth=9,
        cross_layer=4,
        n_ctx=4,
        vision_depth=-1,
        language_depth=-1,
        vision_ctx=-1,
        language_ctx=-1,
        ctx_init='a photo of a',
        ctx_init_sketch='a sketch of a',
        prec='fp32',
        dataset='sketchy',
        gpt_text_file='gpt_file/sketchy_ext.json',
        disable_cross_exchange=False,
        enhance_text=False,
        learn_logit_scale=False,
        eval_mode='category',
        temperature=0.07,
        lambda_cross_modal=1.0,
        lambda_ce=1.0,
        lambda_consistency=1.0,
        lambda_text_consistency=1.0,
        lambda_visual_cross=0.1,
        triplet_margin=0.3,
        prompt_lr=1e-5,
        clip_LN_lr=1e-5,
        weight_decay=1e-4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_model(cfg, classnames, device):
    from src.model_hicropl import CustomCLIP, HiCroPL_SBIR
    from src.utils import load_clip_to_cpu, load_clip_to_cpu_teacher

    clip_model = load_clip_to_cpu(cfg).to(device).float()
    clip_model_frozen = load_clip_to_cpu_teacher(cfg).to(device).float()

    custom_clip = CustomCLIP(cfg, clip_model, clip_model_frozen, classnames=classnames)
    # CustomCLIP đã deepcopy cả 2, bản gốc không còn cần -> giải phóng sớm
    # (nếu không, đỉnh RAM là 6 bản CLIP thay vì 4).
    del clip_model, clip_model_frozen

    lit = HiCroPL_SBIR(cfg=cfg, args=cfg, classnames=classnames, model=custom_clip)
    # LightningModule.print truy cập self.trainer và sẽ raise khi chưa gắn Trainer.
    # configure_optimizers() và _assert_no_param_leak() đều dùng nó.
    lit.print = print
    return lit.to(device)


def make_batch(cfg, n_cls, device, batch_size=2, with_aug=True):
    """Batch giả khớp chữ ký CustomCLIP.forward (len 5 hoặc 7)."""
    def img():
        return torch.randn(batch_size, 3, 224, 224, device=device)

    label = torch.randint(0, n_cls, (batch_size,), device=device)
    filenames = ['fake_%d.jpg' % i for i in range(batch_size)]
    if with_aug:
        return [img(), img(), img(), img(), img(), label, filenames]
    return [img(), img(), img(), label, filenames]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def snapshot(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def diff_names(before, module, atol=0.0):
    """Tên các param đã đổi giá trị so với snapshot."""
    changed = []
    for n, p in module.named_parameters():
        if n not in before:
            continue
        if not torch.allclose(before[n], p.detach(), atol=atol, rtol=0):
            changed.append(n)
    return changed


def layernorm_param_ids(module):
    return {
        id(p)
        for _, m in module.named_modules() if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters(recurse=False)
    }


def is_learner(name):
    return name.startswith(LEARNER_PREFIXES)


def is_distill(name):
    return name.startswith(DISTILL_PREFIXES)


# --------------------------------------------------------------------------
# Checks. Mỗi hàm nhận ctx và trả (ok: bool, detail: str).
# --------------------------------------------------------------------------

CHECKS = []


def check(name, description):
    def deco(fn):
        fn.check_name = name
        fn.description = description
        CHECKS.append(fn)
        return fn
    return deco


@check('teacher_frozen', 'Teacher (clip_distill_*) không có param nào trainable')
def check_teacher_frozen(ctx):
    bad = [n for n, p in ctx.model.named_parameters() if is_distill(n) and p.requires_grad]
    if bad:
        return False, f'{len(bad)} param trainable, vd: {bad[:5]}'
    return True, 'toàn bộ teacher requires_grad=False'


@check('backbone_only_ln', 'Backbone student chỉ trainable ở LayerNorm')
def check_backbone_only_ln(ctx):
    ln_ids = layernorm_param_ids(ctx.model)
    allowed = set()
    if getattr(ctx.model, 'learn_logit_scale', False):
        allowed = {id(ctx.model.clip_photo.logit_scale), id(ctx.model.clip_sketch.logit_scale)}

    bad = [
        n for n, p in ctx.model.named_parameters()
        if p.requires_grad and not is_learner(n) and not is_distill(n)
        and id(p) not in ln_ids and id(p) not in allowed
    ]
    if bad:
        return False, f'{len(bad)} param backbone ngoài LayerNorm, vd: {bad[:5]}'
    return True, 'chỉ LayerNorm trainable'


@check('bare_params_frozen', 'Bare nn.Parameter của CLIP đều đóng băng')
def check_bare_params_frozen(ctx):
    """Check trực diện cho bug gốc: freeze_all_but_bn bỏ sót param không tên
    weight/bias. Riêng logit_scale được miễn nếu learn_logit_scale=True."""
    learn_ls = getattr(ctx.model, 'learn_logit_scale', False)
    bad = []
    for n, p in ctx.model.named_parameters():
        if is_learner(n):
            continue  # learner có MultiheadAttention riêng, đúng là phải trainable
        leaf = n.split('.')[-1]
        if leaf not in CLIP_BARE_PARAMS:
            continue
        if leaf == 'logit_scale' and learn_ls and not is_distill(n):
            continue
        if p.requires_grad:
            bad.append(n)
    if bad:
        return False, f'{len(bad)} bare Parameter còn trainable, vd: {bad[:5]}'
    return True, 'in_proj_*/text_projection/proj/positional_embedding/class_embedding đều frozen'


@check('trainable_budget', f'Tổng param trainable < {MAX_TRAINABLE_PARAMS:,}')
def check_trainable_budget(ctx):
    total = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)
    if total >= MAX_TRAINABLE_PARAMS:
        return False, f'{total:,} params - quá ngưỡng, nhiều khả năng freeze policy thủng'
    return True, f'{total:,} params'


@check('optimizer_matches_trainable', 'Optimizer chứa đúng tập param trainable')
def check_optimizer_matches_trainable(ctx):
    opt = ctx.lit.configure_optimizers()
    in_opt = {id(p) for g in opt.param_groups for p in g['params']}
    trainable = {id(p): n for n, p in ctx.model.named_parameters() if p.requires_grad}

    missing = [n for pid, n in trainable.items() if pid not in in_opt]
    extra = [p for g in opt.param_groups for p in g['params'] if not p.requires_grad]
    if missing:
        return False, f'{len(missing)} param trainable không vào optimizer: {missing[:5]}'
    if extra:
        return False, f'{len(extra)} param trong optimizer nhưng requires_grad=False'
    return True, f'{len(in_opt)} tensors, {len(opt.param_groups)} param groups'


@check('optimizer_no_teacher', 'Không param teacher nào lọt vào optimizer')
def check_optimizer_no_teacher(ctx):
    opt = ctx.lit.configure_optimizers()
    in_opt = {id(p) for g in opt.param_groups for p in g['params']}
    bad = [n for n, p in ctx.model.named_parameters() if is_distill(n) and id(p) in in_opt]
    if bad:
        return False, f'{len(bad)} param teacher trong optimizer: {bad[:5]}'
    return True, 'optimizer sạch teacher'


@check('teacher_eval_mode', 'Teacher ở eval mode kể cả sau model.train()')
def check_teacher_eval_mode(ctx):
    ctx.lit.train()
    problems = []
    if ctx.model.clip_distill_photo.training:
        problems.append('clip_distill_photo.training=True')
    if ctx.model.clip_distill_sketch.training:
        problems.append('clip_distill_sketch.training=True')
    if not ctx.model.clip_photo.training:
        problems.append('clip_photo.training=False (student phải ở train mode)')
    if problems:
        return False, '; '.join(problems)
    return True, 'teacher eval, student train'


@check('forward_no_mutation', 'forward() không mutate param tại chỗ')
def check_forward_no_mutation(ctx):
    """Bản cũ dùng `.data.copy_()` ghi thẳng vào nn.Parameter trong prompt
    learner - param đổi giá trị ngoài optimizer.step()."""
    ctx.lit.train()
    before = snapshot(ctx.model)
    with torch.no_grad():
        ctx.model(ctx.batch, ctx.classnames)
    changed = diff_names(before, ctx.model)
    if changed:
        return False, f'{len(changed)} param bị mutate: {changed[:5]}'
    return True, 'không param nào đổi'


@check('prompt_learner_pure', 'Prompt learner là hàm thuần (2 lần gọi ra cùng kết quả)')
def check_prompt_learner_pure(ctx):
    """Bản cũ ghi đè `.data` mỗi lần forward nên gọi 2 lần liên tiếp cho ra kết
    quả khác nhau dù không có optimizer.step() nào ở giữa."""
    learner = ctx.model.visual_visual_learner
    with torch.no_grad():
        a = learner()
        b = learner()
    for i, (x, y) in enumerate(zip(a[:2], b[:2])):
        if not torch.equal(x, y):
            return False, f'output[{i}] khác nhau giữa 2 lần gọi'
    for i, (xs, ys) in enumerate(zip(a[2:], b[2:])):
        for j, (x, y) in enumerate(zip(xs, ys)):
            if not torch.equal(x, y):
                return False, f'output[{i + 2}][{j}] khác nhau giữa 2 lần gọi'
    return True, 'idempotent'


@check('eval_path_no_mutation', 'extract_eval_features() không mutate param')
def check_eval_path_no_mutation(ctx):
    """Validation gọi learner mỗi batch; nếu learner mutate param thì weights
    trôi trong lúc eval và checkpoint lưu giá trị bẩn."""
    ctx.lit.eval()
    before = snapshot(ctx.model)
    tensor = torch.randn(2, 3, 224, 224, device=ctx.device)
    with torch.no_grad():
        ctx.lit.extract_eval_features(tensor, modality='sketch')
        ctx.lit.extract_eval_features(tensor, modality='photo')
    changed = diff_names(before, ctx.model)
    ctx.lit.train()
    if changed:
        return False, f'{len(changed)} param bị mutate trong eval: {changed[:5]}'
    return True, 'không param nào đổi'


@check('grad_reaches_all_trainable', 'Mọi param trainable đều nhận gradient')
def check_grad_reaches_all_trainable(ctx):
    from src.losses_hicropl import loss_fn_hicropl

    ctx.lit.train()
    ctx.model.zero_grad(set_to_none=True)
    features = ctx.model(ctx.batch, ctx.classnames)
    loss, _ = loss_fn_hicropl(ctx.cfg, features)
    loss.backward()

    no_grad = [n for n, p in ctx.model.named_parameters() if p.requires_grad and p.grad is None]
    ctx.last_loss = float(loss.detach())
    if no_grad:
        return False, f'{len(no_grad)} param trainable không có grad: {no_grad[:8]}'
    return True, f'loss={ctx.last_loss:.4f}, tất cả param trainable có grad'


@check('grad_reaches_knowledge_mapper', 'Knowledge mapper + LKP nhận gradient khác 0')
def check_grad_reaches_knowledge_mapper(ctx):
    """Đây là check quan trọng nhất về mặt khoa học: cơ chế cross-modal knowledge
    flow là đóng góp chính của HiCroPL. Bản dùng `.data.copy_()` cắt graph nên
    các mạng này có grad=None và đứng yên ở giá trị init suốt quá trình train."""
    from src.losses_hicropl import loss_fn_hicropl

    ctx.lit.train()
    ctx.model.zero_grad(set_to_none=True)
    features = ctx.model(ctx.batch, ctx.classnames)
    loss, _ = loss_fn_hicropl(ctx.cfg, features)
    loss.backward()

    vv = ctx.model.visual_visual_learner
    if getattr(vv, 'disable_cross_exchange', False):
        return True, 'skipped (disable_cross_exchange=True)'

    targets = [
        ('photo2sketch_net', vv.photo2sketch_net.parameters()),
        ('sketch2photo_net', vv.sketch2photo_net.parameters()),
        ('attn_pooling_photo_nets', vv.attn_pooling_photo_nets.parameters()),
        ('attn_pooling_sketch_nets', vv.attn_pooling_sketch_nets.parameters()),
        ('photo_proxy_tokens', vv.photo_proxy_tokens.parameters()),
        ('sketch_proxy_tokens', vv.sketch_proxy_tokens.parameters()),
    ]
    dead = []
    for name, params in targets:
        grads = [p.grad for p in params]
        if not any(g is not None and torch.any(g != 0) for g in grads):
            dead.append(name)
    if dead:
        return False, f'không nhận gradient (graph bị cắt): {dead}'
    return True, 'cả 6 nhóm đều có gradient khác 0'


@check('teacher_no_grad', 'Teacher không tích luỹ gradient')
def check_teacher_no_grad(ctx):
    from src.losses_hicropl import loss_fn_hicropl

    ctx.lit.train()
    ctx.model.zero_grad(set_to_none=True)
    features = ctx.model(ctx.batch, ctx.classnames)
    loss, _ = loss_fn_hicropl(ctx.cfg, features)
    loss.backward()

    bad = [n for n, p in ctx.model.named_parameters() if is_distill(n) and p.grad is not None]
    if bad:
        return False, f'{len(bad)} param teacher có grad: {bad[:5]}'
    return True, 'teacher hoàn toàn ngoài graph'


@check('step_updates_prompts_only', 'optimizer.step() chỉ đổi param trainable')
def check_step_updates_prompts_only(ctx):
    from src.losses_hicropl import loss_fn_hicropl

    ctx.lit.train()
    opt = ctx.lit.configure_optimizers()
    # lr lớn để một step tạo thay đổi đo được
    for g in opt.param_groups:
        g['lr'] = 1e-2

    before = snapshot(ctx.model)
    ctx.model.zero_grad(set_to_none=True)
    features = ctx.model(ctx.batch, ctx.classnames)
    loss, _ = loss_fn_hicropl(ctx.cfg, features)
    loss.backward()
    opt.step()

    changed = set(diff_names(before, ctx.model))
    frozen_changed = [
        n for n, p in ctx.model.named_parameters()
        if not p.requires_grad and n in changed
    ]
    if frozen_changed:
        return False, f'{len(frozen_changed)} param frozen bị đổi: {frozen_changed[:5]}'
    if not changed:
        return False, 'không param nào đổi sau step - optimizer không tác dụng'
    return True, f'{len(changed)} param trainable được cập nhật, 0 param frozen bị đụng'


@check('logit_scale_clamped', 'logit_scale được clamp <= 100')
def check_logit_scale_clamped(ctx):
    with torch.no_grad():
        original = ctx.model.clip_photo.logit_scale.detach().clone()
        ctx.model.clip_photo.logit_scale.fill_(50.0)  # exp(50) = vô cực thực tế
        scaled = ctx.model._exp_logit_scale(ctx.model.clip_photo.logit_scale)
        ctx.model.clip_photo.logit_scale.copy_(original)
    if not torch.isfinite(scaled) or scaled.item() > 100.0 + 1e-3:
        return False, f'exp(logit_scale)={scaled.item()} - clamp không hoạt động'
    return True, f'clamp về {scaled.item():.1f}'


@check('assert_hook_runs', '_assert_no_param_leak() chạy sạch trên model thật')
def check_assert_hook_runs(ctx):
    try:
        ctx.lit._assert_no_param_leak()
    except RuntimeError as e:
        return False, str(e).replace('\n', ' | ')
    return True, 'không phát hiện leak'


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

if sys.stdout.isatty() and os.environ.get('NO_COLOR') is None:
    GREEN, RED, DIM, RESET = '\033[32m', '\033[31m', '\033[2m', '\033[0m'
else:
    GREEN = RED = DIM = RESET = ''


def run_config(label, cfg_overrides, device, classnames, verbose):
    print(f'\n{"=" * 78}\nCONFIG: {label}')
    if cfg_overrides:
        print(f'  {cfg_overrides}')
    print('=' * 78)

    with_aug = not cfg_overrides.pop('_no_aug', False)
    cfg = build_cfg(**cfg_overrides)

    try:
        lit = build_model(cfg, classnames, device)
    except Exception:
        print(f'{RED}BUILD FAILED{RESET}')
        traceback.print_exc()
        return [(label, 'build_model', False, 'exception khi dựng model')]

    ctx = SimpleNamespace(
        cfg=cfg, lit=lit, model=lit.model, device=device,
        classnames=classnames, last_loss=None,
        batch=make_batch(cfg, len(classnames), device, with_aug=with_aug),
    )

    results = []
    for fn in CHECKS:
        try:
            ok, detail = fn(ctx)
        except Exception as e:
            ok, detail = False, f'exception: {type(e).__name__}: {e}'
            if verbose:
                traceback.print_exc()
        results.append((label, fn.check_name, ok, detail))
        mark = f'{GREEN}PASS{RESET}' if ok else f'{RED}FAIL{RESET}'
        print(f'  [{mark}] {fn.check_name:<32} {DIM}{detail}{RESET}')

    del lit, ctx
    if device == 'cuda':
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device',
                    default='cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu')
    ap.add_argument('--all', action='store_true', help='Quét mọi cấu hình ablation')
    ap.add_argument('--nclass', type=int, default=4, help='Số classname giả lập')
    ap.add_argument('--list', action='store_true', help='In danh sách check rồi thoát')
    ap.add_argument('-v', '--verbose', action='store_true', help='In traceback đầy đủ')
    args = ap.parse_args()

    if args.list:
        print(f'{len(CHECKS)} checks:')
        for fn in CHECKS:
            print(f'  {fn.check_name:<34} {fn.description}')
        return 0

    if torch is None:
        print('Cần cài torch để chạy check (chỉ --list là không cần).', file=sys.stderr)
        return 1

    classnames = ['airplane', 'alarm_clock', 'ant', 'apple'][:args.nclass]
    print(f'Device: {args.device} | classnames: {classnames}')

    configs = [('default', {})]
    if args.all:
        configs += [
            ('disable_cross_exchange', {'disable_cross_exchange': True}),
            ('enhance_text', {'enhance_text': True}),
            ('learn_logit_scale', {'learn_logit_scale': True}),
            ('no_augmentation', {'_no_aug': True}),
            ('shallow_prompt', {'prompt_depth': 3, 'cross_layer': 1}),
        ]

    all_results = []
    for label, overrides in configs:
        all_results += run_config(label, overrides, args.device, classnames, args.verbose)

    failed = [r for r in all_results if not r[2]]
    print(f'\n{"=" * 78}')
    print(f'TỔNG: {len(all_results) - len(failed)}/{len(all_results)} PASS')
    if failed:
        print(f'{RED}FAIL:{RESET}')
        for label, name, _, detail in failed:
            print(f'  [{label}] {name}: {detail}')
        return 1
    print(f'{GREEN}Tất cả bất biến đều giữ.{RESET}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
