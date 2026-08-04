"""Tests cho src/run_logging.py.

Phần CSV là pure-Python nên chạy được cả khi chưa cài torch/lightning (tự stub).
Phần callback cần torch để đếm param, sẽ tự skip nếu thiếu.
"""

import csv
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytorch_lightning  # noqa: F401
except ImportError:
    _pl = types.ModuleType('pytorch_lightning')
    _pl.Callback = object
    sys.modules['pytorch_lightning'] = _pl

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.run_logging import (  # noqa: E402
    RunCSVLogger, append_csv_row, collect_metrics, make_run_dir,
    save_config_snapshot, write_csv, _to_scalar,
)


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


class TestCSVHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_append_same_schema(self):
        p = os.path.join(self.tmp, 's.csv')
        append_csv_row(p, {'run_id': 'r1', 'best_value': 0.5})
        append_csv_row(p, {'run_id': 'r2', 'best_value': 0.6})
        rows = read_csv(p)
        self.assertEqual([r['run_id'] for r in rows], ['r1', 'r2'])

    def test_append_new_column_keeps_old_rows(self):
        """Run sau thêm hyperparameter mới thì cột mới không được làm mất dòng cũ."""
        p = os.path.join(self.tmp, 's.csv')
        append_csv_row(p, {'run_id': 'r1', 'best_value': 0.5})
        append_csv_row(p, {'run_id': 'r2', 'best_value': 0.6, 'cfg_new_flag': True})
        rows = read_csv(p)
        self.assertEqual(len(rows), 2)
        self.assertIn('cfg_new_flag', rows[0])
        self.assertEqual(rows[0]['cfg_new_flag'], '', 'dòng cũ phải trống ở cột mới')
        self.assertEqual(rows[1]['cfg_new_flag'], 'True')

    def test_append_missing_column_does_not_shift(self):
        """Row thiếu cột phải để trống chứ không đẩy lệch các cột sau."""
        p = os.path.join(self.tmp, 's.csv')
        append_csv_row(p, {'a': 1, 'b': 2, 'c': 3})
        append_csv_row(p, {'a': 9, 'c': 7})
        rows = read_csv(p)
        self.assertEqual(rows[1]['a'], '9')
        self.assertEqual(rows[1]['b'], '')
        self.assertEqual(rows[1]['c'], '7')

    def test_write_csv_overwrites_and_unions_keys(self):
        p = os.path.join(self.tmp, 'e.csv')
        write_csv(p, [{'epoch': 0, 'mAP': 0.1}, {'epoch': 1, 'mAP': 0.2, 'loss_ce': 3.0}])
        rows = read_csv(p)
        self.assertEqual(list(rows[0].keys()), ['epoch', 'mAP', 'loss_ce'])
        self.assertEqual(rows[0]['loss_ce'], '')

        write_csv(p, [{'epoch': 0, 'mAP': 0.9}])
        rows = read_csv(p)
        self.assertEqual(len(rows), 1, 'write_csv phải ghi đè chứ không append')

    def test_write_csv_empty_is_noop(self):
        p = os.path.join(self.tmp, 'empty.csv')
        write_csv(p, [])
        self.assertFalse(os.path.exists(p))

    def test_to_scalar(self):
        class FakeTensor:
            def item(self):
                return 0.123456789

        self.assertEqual(_to_scalar(FakeTensor()), 0.123457)
        self.assertEqual(_to_scalar(2), 2.0)
        self.assertEqual(_to_scalar('n/a'), 'n/a')

    def test_make_run_dir_isolates_runs(self):
        """Chạy lại cùng exp_name phải ra thư mục khác, không đè log cũ."""
        log_dir = os.path.join(self.tmp, 'logs')
        d1, id1 = make_run_dir(log_dir, 'exp', 'run-a')
        d2, id2 = make_run_dir(log_dir, 'exp', 'run-b')
        self.assertNotEqual(d1, d2)
        self.assertEqual((id1, id2), ('run-a', 'run-b'))
        self.assertTrue(os.path.isdir(d1) and os.path.isdir(d2))
        # cùng exp_name -> chung thư mục cha
        self.assertEqual(os.path.dirname(d1), os.path.dirname(d2))

    def test_make_run_dir_default_id_is_timestamp(self):
        d, rid = make_run_dir(os.path.join(self.tmp, 'logs'), 'exp')
        self.assertRegex(rid, r'^\d{8}-\d{6}$')
        self.assertTrue(d.endswith(rid))

    def test_config_snapshot_handles_unserialisable(self):
        cfg = types.SimpleNamespace(lr=1e-3, name='x', weird=object(), flag=True)
        d, _ = make_run_dir(os.path.join(self.tmp, 'logs'), 'exp', 'r1')
        path = save_config_snapshot(d, cfg)

        import json
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['lr'], 1e-3)
        self.assertEqual(data['flag'], True)
        self.assertIsInstance(data['weird'], str, 'giá trị không serialise được phải thành str')


class FakeTrainer:
    def __init__(self, metrics, epoch=0, step=0, sanity=False, optimizers=()):
        self.callback_metrics = metrics
        self.current_epoch = epoch
        self.global_step = step
        self.sanity_checking = sanity
        self.optimizers = list(optimizers)


class FakeOptimizer:
    def __init__(self, lrs):
        self.param_groups = [{'lr': lr} for lr in lrs]


class TestSummaryLogic(unittest.TestCase):
    """Phần không cần torch: chọn best epoch, lọc metric, ghi dòng summary.

    `_write_summary` chỉ gọi `_param_counts` khi chưa có `_counts`, nên set sẵn
    thuộc tính đó là chạy được mà không cần model thật.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.summary = os.path.join(self.tmp, 'runs_summary.csv')
        cfg = types.SimpleNamespace(eval_mode='category', dataset='sketchy',
                                    save_dir=os.path.join(self.tmp, 'ckpt'))
        self.run_dir, _ = make_run_dir(os.path.join(self.tmp, 'logs'), 'exp1', 'r-test')
        self.cb = RunCSVLogger(cfg, self.run_dir, 'exp1', self.summary)
        self.cb._counts = {'total': 100, 'trainable': 10, 'prompt': 6, 'layernorm': 4}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_metrics_dropped(self):
        metrics = {'train_loss': 1.0, 'train_loss_step': 9.9, 'train_loss_epoch': 1.0}
        out = collect_metrics(FakeTrainer(metrics))
        self.assertNotIn('train_loss_step', out)
        self.assertIn('train_loss_epoch', out)

    def test_best_row_picks_max_not_last(self):
        self.cb.rows = [
            {'epoch': 0, 'mAP': 0.40},
            {'epoch': 1, 'mAP': 0.62},
            {'epoch': 2, 'mAP': 0.55},
        ]
        row, val = self.cb._best_row()
        self.assertEqual(val, 0.62)
        self.assertEqual(row['epoch'], 1)

    def test_best_row_ignores_non_numeric(self):
        self.cb.rows = [{'epoch': 0, 'mAP': 'nan-ish'}, {'epoch': 1, 'mAP': 0.3}]
        row, val = self.cb._best_row()
        self.assertEqual(val, 0.3)
        self.assertEqual(row['epoch'], 1)

    def test_best_row_empty(self):
        self.assertEqual(self.cb._best_row(), (None, None))

    def test_monitor_defaults_by_eval_mode(self):
        self.assertEqual(self.cb.monitor, 'mAP')
        fg_cfg = types.SimpleNamespace(eval_mode='fine_grained')
        fg = RunCSVLogger(fg_cfg, self.tmp, 'e', self.summary)  # run_dir bat ky
        self.assertEqual(fg.monitor, 'top1')

    def test_summary_row_content(self):
        self.cb.rows = [{'epoch': 0, 'mAP': 0.4, 'train_loss': 2.0},
                        {'epoch': 1, 'mAP': 0.7, 'train_loss': 1.1}]
        self.cb._write_summary(None, None, status='completed')

        rows = read_csv(self.summary)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['best_value'], '0.7')
        self.assertEqual(rows[0]['best_epoch'], '1')
        self.assertEqual(rows[0]['best_train_loss'], '1.1',
                         'metric khác của epoch tốt nhất cũng phải được chụp')
        self.assertEqual(rows[0]['trainable_params'], '10')
        self.assertEqual(rows[0]['cfg_dataset'], 'sketchy')

    def test_pl_overwrites_log_attribute(self):
        """PL gán `callback.log = lightning_module.log` cho mọi callback.

        Nếu logger nội bộ đặt tên `self.log` thì nó bị ghi đè bằng một function
        và mọi hook chết với AttributeError. Logger phải nằm ở tên khác.
        """
        self.assertIsInstance(self.cb._logger, logging.Logger)

        def fake_pl_log(*a, **kw):  # đúng thứ PL gán vào
            pass

        self.cb.log = fake_pl_log  # PL làm việc này trước on_fit_start
        self.assertIsInstance(self.cb._logger, logging.Logger,
                              'logger nội bộ không được dùng chung tên với PL')

        # Hook vẫn phải chạy bình thường sau khi PL ghi đè .log
        self.cb.rows = [{'epoch': 0, 'mAP': 0.5}]
        self.cb._write_summary(None, None, status='completed')
        self.assertEqual(len(read_csv(self.summary)), 1)

    def test_never_fail_survives_broken_logger(self):
        """Handler lỗi không được dựa vào attribute có thể chính là nguyên nhân."""
        self.cb._logger = 'not-a-logger'
        self.cb.run_dir = '\x00invalid'
        self.cb.on_fit_start(FakeTrainer({}), None)  # không được raise

    def test_two_runs_share_summary_file(self):
        self.cb.rows = [{'epoch': 0, 'mAP': 0.4}]
        self.cb._write_summary(None, None, status='completed')

        cb2_dir, _ = make_run_dir(os.path.join(self.tmp, 'logs'), 'exp2')
        cb2 = RunCSVLogger(self.cb.cfg, cb2_dir, 'exp2', self.summary)
        cb2._counts = self.cb._counts
        cb2.rows = [{'epoch': 0, 'mAP': 0.8}]
        cb2._write_summary(None, None, status='completed')

        rows = read_csv(self.summary)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['exp_name'] for r in rows], ['exp1', 'exp2'])


@unittest.skipUnless(HAS_TORCH, 'cần torch để đếm param')
class TestRunCSVLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = types.SimpleNamespace(
            eval_mode='category', dataset='sketchy', backbone='ViT-B/32',
            epochs=3, batch_size=8, prompt_lr=1e-5, clip_LN_lr=1e-5,
            save_dir=os.path.join(self.tmp, 'ckpt'),
        )

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(4)
                self.lin = nn.Linear(4, 4)
                self.lin.weight.requires_grad_(False)
                self.lin.bias.requires_grad_(False)

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Tiny()

        self.module = Wrapper()
        self.summary = os.path.join(self.tmp, 'runs_summary.csv')
        self.run_dir, self.run_id = make_run_dir(os.path.join(self.tmp, 'logs'), 'exp1')
        self.cb = RunCSVLogger(self.cfg, self.run_dir, 'exp1', self.summary,
                               run_id=self.run_id, log_every_n_steps=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _epoch(self, trainer):
        self.cb.on_train_epoch_start(trainer, self.module)
        self.cb.on_validation_epoch_end(trainer, self.module)

    def test_epoch_csv_and_summary(self):
        opt = FakeOptimizer([1e-5, 2e-5])
        self.cb.on_fit_start(FakeTrainer({}), self.module)

        self._epoch(FakeTrainer({'mAP': 0.40, 'train_loss': 2.0}, epoch=0, optimizers=[opt]))
        self._epoch(FakeTrainer({'mAP': 0.62, 'train_loss': 1.2}, epoch=1, optimizers=[opt]))
        self._epoch(FakeTrainer({'mAP': 0.55, 'train_loss': 1.0}, epoch=2, optimizers=[opt]))

        rows = read_csv(self.cb.epoch_csv)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['epoch'] for r in rows], ['0', '1', '2'])
        self.assertEqual(rows[0]['lr_prompt'], '1e-05')
        self.assertEqual(rows[0]['lr_ln'], '2e-05')

        self.cb.on_fit_end(FakeTrainer({}), self.module)
        summary = read_csv(self.summary)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]['status'], 'completed')
        self.assertEqual(summary[0]['best_value'], '0.62')
        self.assertEqual(summary[0]['best_epoch'], '1', 'phải chọn epoch tốt nhất, không phải cuối')
        self.assertEqual(summary[0]['epochs_done'], '3')
        self.assertEqual(summary[0]['cfg_dataset'], 'sketchy')

    def test_train_steps_csv_respects_interval(self):
        """log_every_n_steps=2 -> chỉ ghi ở step 0, 2, 4."""
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        for step in range(6):
            self.cb.on_train_batch_end(
                FakeTrainer({'train_loss_step': 1.0 / (step + 1)}, epoch=0, step=step),
                self.module, None, None, step)

        rows = read_csv(self.cb.step_csv)
        self.assertEqual([r['global_step'] for r in rows], ['0', '2', '4'])
        self.assertEqual(self.cb._n_step_rows, 3)
        # hậu tố _step bị lược bỏ để tên cột khớp với metrics_epoch.csv
        self.assertIn('train_loss', rows[0])
        self.assertNotIn('train_loss_step', rows[0])

    def test_train_steps_disabled(self):
        self.cb.log_every_n_steps = 0
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self.cb.on_train_batch_end(FakeTrainer({'train_loss_step': 1.0}), self.module, None, None, 0)
        self.assertFalse(os.path.exists(self.cb.step_csv))

    def test_run_dir_contains_all_artifacts(self):
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self.cb.on_train_batch_end(FakeTrainer({'train_loss_step': 1.0}), self.module, None, None, 0)
        self._epoch(FakeTrainer({'mAP': 0.5}, epoch=0))
        self.cb.on_fit_end(FakeTrainer({}), self.module)

        for name in ('config.json', 'train_steps.csv', 'metrics_epoch.csv'):
            self.assertTrue(os.path.exists(os.path.join(self.run_dir, name)), name)
        # summary nằm NGOÀI run_dir
        self.assertFalse(self.summary.startswith(self.run_dir))
        self.assertTrue(os.path.exists(self.summary))

    def test_rerun_same_exp_does_not_overwrite(self):
        """Hai run cùng exp_name phải có metrics_epoch.csv riêng."""
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self._epoch(FakeTrainer({'mAP': 0.11}, epoch=0))

        dir2, id2 = make_run_dir(os.path.join(self.tmp, 'logs'), 'exp1', 'run-2')
        cb2 = RunCSVLogger(self.cfg, dir2, 'exp1', self.summary, run_id=id2)
        cb2.on_fit_start(FakeTrainer({}), self.module)
        cb2._epoch = self._epoch
        cb2.on_train_epoch_start(FakeTrainer({}), self.module)
        cb2.on_validation_epoch_end(FakeTrainer({'mAP': 0.99}, epoch=0), self.module)

        self.assertNotEqual(self.cb.epoch_csv, cb2.epoch_csv)
        self.assertEqual(read_csv(self.cb.epoch_csv)[0]['mAP'], '0.11',
                         'run đầu bị run sau ghi đè')
        self.assertEqual(read_csv(cb2.epoch_csv)[0]['mAP'], '0.99')

    def test_sanity_check_epoch_skipped(self):
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self._epoch(FakeTrainer({'mAP': 0.01}, epoch=0, sanity=True))
        self.assertEqual(self.cb.rows, [], 'sanity check không được ghi thành epoch')

    def test_exception_writes_failed_summary(self):
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self._epoch(FakeTrainer({'mAP': 0.3}, epoch=0))
        self.cb.on_exception(FakeTrainer({}), self.module, ValueError('boom'))

        summary = read_csv(self.summary)
        self.assertEqual(len(summary), 1)
        self.assertTrue(summary[0]['status'].startswith('failed:ValueError'))
        self.assertEqual(summary[0]['best_value'], '0.3',
                         'run hỏng vẫn phải giữ kết quả các epoch đã chạy')

    def test_summary_written_once(self):
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self._epoch(FakeTrainer({'mAP': 0.3}, epoch=0))
        self.cb.on_exception(FakeTrainer({}), self.module, ValueError('boom'))
        self.cb.on_fit_end(FakeTrainer({}), self.module)
        self.assertEqual(len(read_csv(self.summary)), 1, 'không được ghi 2 dòng cho 1 run')

    def test_param_counts_split(self):
        counts = self.cb._param_counts(self.module)
        self.assertEqual(counts['trainable'], 8, 'chỉ LayerNorm weight+bias (4+4)')
        self.assertEqual(counts['layernorm'], 8)
        self.assertEqual(counts['total'], 8 + 16 + 4)

    def test_hook_errors_do_not_propagate(self):
        """Lỗi logging không được giết run training."""
        self.cb.epoch_csv = os.path.join(self.tmp, 'nonexistent\x00bad', 'x.csv')
        self.cb.on_fit_start(FakeTrainer({}), self.module)
        self._epoch(FakeTrainer({'mAP': 0.3}, epoch=0))  # không được raise


if __name__ == '__main__':
    unittest.main()
