"""Ghi kết quả train ra CSV + text log để theo dõi và đối chiếu giữa các run.

Ba nơi ghi:
  <log_dir>/<exp_name>/metrics_epoch.csv  - chi tiết từng epoch của run này
  <log_dir>/<exp_name>/run.log            - text log: config, param count, tiến độ
  <summary_csv>                           - 1 dòng/run, NẰM NGOÀI log_dir để so
                                            sánh các run với nhau

CSV per-epoch được ghi lại toàn bộ sau mỗi epoch (không append) nên schema tự
mở rộng khi có metric mới, và crash giữa chừng vẫn còn dữ liệu các epoch trước.
Summary CSV thì append, có xử lý trường hợp file cũ thiếu cột.
"""

import csv
import logging
import os
import time
from datetime import datetime

from pytorch_lightning import Callback


# --------------------------------------------------------------------------
# Text log
# --------------------------------------------------------------------------

def setup_run_logger(log_dir, exp_name, filename='run.log'):
    """Logger ghi đồng thời ra console và <log_dir>/<exp_name>/run.log."""
    run_dir = os.path.join(log_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, filename)

    logger = logging.getLogger(f'hicropl.{exp_name}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Gọi lại nhiều lần (vd. trong test) không được nhân đôi handler
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    fmt = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(sh)

    logger.info('=' * 70)
    logger.info('RUN START %s | exp=%s', datetime.now().isoformat(timespec='seconds'), exp_name)
    return logger


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------

def write_csv(path, rows, fieldnames=None):
    """Ghi đè toàn bộ CSV. fieldnames mặc định là hợp của mọi key trong rows."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def append_csv_row(path, row):
    """Append 1 dòng, tự xử lý khi file cũ có schema khác.

    Nếu row có cột mà file cũ không có, đọc lại toàn bộ file và ghi lại với
    header hợp nhất - tránh việc cột mới bị âm thầm rơi mất.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        write_csv(path, [row])
        return

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        new_fields = [k for k in row if k not in existing_fields]
        old_rows = list(reader) if new_fields else None

    if new_fields:
        write_csv(path, old_rows + [row], fieldnames=existing_fields + new_fields)
        return

    with open(path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=existing_fields, extrasaction='ignore').writerow(row)


def _to_scalar(v):
    """Tensor/np scalar -> float; giữ nguyên nếu không chuyển được."""
    try:
        if hasattr(v, 'item'):
            return round(float(v.item()), 6)
        return round(float(v), 6)
    except (TypeError, ValueError, RuntimeError):
        return v


def collect_metrics(trainer):
    """callback_metrics -> dict phẳng, bỏ bản `*_step` cho gọn."""
    out = {}
    for k, v in trainer.callback_metrics.items():
        if k.endswith('_step'):
            continue
        out[k] = _to_scalar(v)
    return out


# --------------------------------------------------------------------------
# Callback
# --------------------------------------------------------------------------

_FALLBACK_LOG = logging.getLogger('hicropl.run_logging')


def _never_fail(fn):
    """Lỗi trong callback logging không được giết một run training 10 tiếng.

    Ghi lại lỗi rồi đi tiếp. Chỉ dùng cho hook - các hàm CSV bên dưới vẫn raise
    bình thường để test bắt được.

    Handler KHÔNG được đụng tới attribute của instance: nếu chính attribute đó
    là nguyên nhân lỗi thì handler cũng nổ và một warning biến thành chuỗi crash.
    """
    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception as e:
            log = getattr(self, '_logger', None)
            if not isinstance(log, logging.Logger):
                log = _FALLBACK_LOG
            log.warning('RunCSVLogger.%s lỗi (bỏ qua): %s: %s',
                        fn.__name__, type(e).__name__, e, exc_info=True)
            return None
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


class RunCSVLogger(Callback):
    """Ghi metrics từng epoch ra CSV, và 1 dòng tổng kết ra CSV chung.

    Args:
        cfg: opts namespace (để chụp lại hyperparameter vào dòng summary)
        log_dir, exp_name: xác định <log_dir>/<exp_name>/
        summary_csv: đường dẫn CSV chung, NGOÀI log_dir
        logger: logging.Logger từ setup_run_logger()
        monitor: tên metric để chọn epoch tốt nhất. None = tự đoán theo eval_mode
    """

    # Hyperparameter được chụp vào dòng summary để so sánh giữa các run
    TRACKED_OPTS = (
        'dataset', 'backbone', 'eval_mode', 'epochs', 'batch_size',
        'prompt_lr', 'clip_LN_lr', 'weight_decay',
        'n_ctx', 'prompt_depth', 'cross_layer',
        'disable_cross_exchange', 'disable_augmentation', 'enhance_text',
        'learn_logit_scale',
        'temperature', 'lambda_cross_modal', 'lambda_ce', 'lambda_consistency',
        'lambda_text_consistency', 'lambda_visual_cross',
    )

    def __init__(self, cfg, log_dir, exp_name, summary_csv, logger=None, monitor=None):
        super().__init__()
        self.cfg = cfg
        self.exp_name = exp_name
        self.run_dir = os.path.join(log_dir, exp_name)
        self.epoch_csv = os.path.join(self.run_dir, 'metrics_epoch.csv')
        self.summary_csv = summary_csv
        # KHÔNG đặt tên `self.log`: PyTorch Lightning gán
        # `callback.log = lightning_module.log` cho mọi callback trước khi
        # chạy hook, nên attribute đó sẽ bị ghi đè bằng hàm log metric của PL.
        self._logger = logger or logging.getLogger(f'hicropl.{exp_name}')

        if monitor is None:
            monitor = 'top1' if getattr(cfg, 'eval_mode', 'category') == 'fine_grained' else 'mAP'
        self.monitor = monitor

        self.rows = []
        self.run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._t_start = None
        self._t_epoch = None
        self._summary_written = False

    # -- hooks -------------------------------------------------------------

    @_never_fail
    def on_fit_start(self, trainer, pl_module):
        self._t_start = time.time()
        os.makedirs(self.run_dir, exist_ok=True)

        self._logger.info('run_id=%s | checkpoint=%s', self.run_id,
                      os.path.abspath(getattr(self.cfg, 'save_dir', 'saved_models')))
        self._logger.info('Config: %s', {k: getattr(self.cfg, k, None) for k in self.TRACKED_OPTS})

        counts = self._param_counts(pl_module)
        self._logger.info(
            'Params: trainable=%s / total=%s (%.3f%%) | prompt=%s | layernorm=%s',
            f"{counts['trainable']:,}", f"{counts['total']:,}",
            100.0 * counts['trainable'] / max(counts['total'], 1),
            f"{counts['prompt']:,}", f"{counts['layernorm']:,}",
        )
        self._counts = counts

    @_never_fail
    def on_train_epoch_start(self, trainer, pl_module):
        self._t_epoch = time.time()

    @_never_fail
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        row = {
            'run_id': self.run_id,
            'exp_name': self.exp_name,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'epoch_time_s': round(time.time() - self._t_epoch, 1) if self._t_epoch else '',
        }
        row.update(self._learning_rates(trainer))
        row.update(collect_metrics(trainer))
        self.rows.append(row)

        # Ghi lại toàn bộ file mỗi epoch: schema tự mở rộng, crash vẫn còn dữ liệu
        write_csv(self.epoch_csv, self.rows)

        self._logger.info('epoch %d | %s', trainer.current_epoch, self._fmt_row(row))

    @_never_fail
    def on_fit_end(self, trainer, pl_module):
        self._write_summary(trainer, pl_module, status='completed')

    @_never_fail
    def on_exception(self, trainer, pl_module, exception):
        self._logger.error('Run hỏng: %s: %s', type(exception).__name__, exception)
        self._write_summary(trainer, pl_module, status=f'failed:{type(exception).__name__}')

    # -- internals ---------------------------------------------------------

    def _write_summary(self, trainer, pl_module, status):
        if self._summary_written:
            return
        self._summary_written = True

        best_row, best_val = self._best_row()
        duration_min = round((time.time() - self._t_start) / 60.0, 2) if self._t_start else ''

        summary = {
            'run_id': self.run_id,
            'exp_name': self.exp_name,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'status': status,
            'epochs_done': len(self.rows),
            'duration_min': duration_min,
            'monitor': self.monitor,
            'best_value': best_val if best_val is not None else '',
            'best_epoch': best_row.get('epoch', '') if best_row else '',
        }
        # Mọi metric của epoch tốt nhất, prefix best_ để không đụng tên cột config
        if best_row:
            for k, v in best_row.items():
                if k in ('run_id', 'exp_name', 'timestamp', 'epoch', 'global_step'):
                    continue
                summary[f'best_{k}'] = v

        counts = getattr(self, '_counts', None) or self._param_counts(pl_module)
        summary['trainable_params'] = counts['trainable']
        summary['total_params'] = counts['total']

        for k in self.TRACKED_OPTS:
            summary[f'cfg_{k}'] = getattr(self.cfg, k, '')
        summary['ckpt_dir'] = os.path.abspath(
            os.path.join(getattr(self.cfg, 'save_dir', 'saved_models'), self.exp_name))
        summary['epoch_csv'] = os.path.abspath(self.epoch_csv)

        append_csv_row(self.summary_csv, summary)

        self._logger.info('%s | best %s=%s @epoch %s | %s phút',
                      status, self.monitor, summary['best_value'],
                      summary['best_epoch'], duration_min)
        self._logger.info('Per-epoch CSV : %s', os.path.abspath(self.epoch_csv))
        self._logger.info('Summary CSV   : %s', os.path.abspath(self.summary_csv))

    def _best_row(self):
        """Epoch có monitor cao nhất (mọi metric ở đây đều là higher-is-better)."""
        best_row, best_val = None, None
        for r in self.rows:
            v = r.get(self.monitor)
            if not isinstance(v, (int, float)):
                continue
            if best_val is None or v > best_val:
                best_row, best_val = r, v
        return best_row, best_val

    @staticmethod
    def _learning_rates(trainer):
        out = {}
        for oi, opt in enumerate(trainer.optimizers or []):
            for gi, g in enumerate(opt.param_groups):
                key = 'lr_prompt' if gi == 0 else ('lr_ln' if gi == 1 else f'lr_g{gi}')
                if oi > 0:
                    key = f'opt{oi}_{key}'
                out[key] = g.get('lr', '')
        return out

    @staticmethod
    def _param_counts(pl_module):
        import torch

        inner = getattr(pl_module, 'model', pl_module)
        total = trainable = prompt = layernorm = 0
        learner_prefixes = ('visual_visual_learner', 'text_prompt_photo', 'text_prompt_sketch')
        ln_ids = {
            id(p)
            for _, m in inner.named_modules() if isinstance(m, torch.nn.LayerNorm)
            for p in m.parameters(recurse=False)
        }
        for n, p in inner.named_parameters():
            total += p.numel()
            if not p.requires_grad:
                continue
            trainable += p.numel()
            if n.startswith(learner_prefixes):
                prompt += p.numel()
            elif id(p) in ln_ids:
                layernorm += p.numel()
        return {'total': total, 'trainable': trainable,
                'prompt': prompt, 'layernorm': layernorm}

    def _fmt_row(self, row):
        skip = {'run_id', 'exp_name', 'timestamp', 'epoch', 'global_step'}
        parts = []
        for k, v in row.items():
            if k in skip:
                continue
            parts.append(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}')
        return ' '.join(parts)
