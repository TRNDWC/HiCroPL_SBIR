"""Read-only gradient/parameter-norm diagnostic logger for HiCroPL-SBIR training.

Defines `GradLogCallback`, a `pytorch_lightning.Callback` that hooks EXACTLY
ONE Lightning callback method (`on_before_optimizer_step`) to inspect
`.grad`/`.data` on the model's trainable parameters every N-th optimizer step
(default 50) and append one CSV row per parameter group:

    step, epoch, group, grad_l2, param_l2, ratio

group is one of {text_prompt, visual_prompt, mapper_lkp, layernorm} -- the
SAME four groups from the project audit's Q4 answer:
    (a) text_prompt   = text_prompt_photo + text_prompt_sketch
    (b) visual_prompt = visual_visual_learner.cross_prompts_photo / _sketch
                        (== ctx_photo/ctx_sketch, same tensors, deduped)
    (c) mapper_lkp    = everything else inside visual_visual_learner
                        (photo2sketch_net, attn_pooling_photo_nets,
                        photo_proxy_token, sketch2photo_net,
                        attn_pooling_sketch_nets, sketch_proxy_token,
                        free_source, ln_selfrefine -- whichever exist)
    (d) layernorm     = LayerNorm parameters inside model.clip (the frozen
                        backbone), identified by module membership exactly
                        like freeze_all_but_bn() does (src/model_hicropl.py)

This callback is READ-ONLY with respect to training: it never touches
`.grad`, `.data`, the optimizer, the loss, or any hyperparameter. It only
reads already-computed gradients (populated by backward(), before
optimizer.step() runs) and writes to its own CSV file. It performs NO
sampling/inference itself -- it observes whatever step is already running.

If a group receives grad_l2 == 0 (or every param's .grad is None) at EVERY
logged step throughout the whole run, a warning is printed to stdout at
on_fit_end:
    "[DEAD] group=<name> nhận gradient bằng 0"

Coverage is enforced at setup(): every trainable (requires_grad=True)
parameter under the model must fall into exactly one of the four groups, or
a RuntimeError is raised naming the unclassified parameter -- this tool only
supports the default architecture (VisualVisualPromptLearner +
SimpleTextPromptLearner x2, i.e. cfg.no_prompt_learning=False and
cfg.use_text_visual_exchange=False); it refuses to guess for the other two
architectures rather than silently mis-grouping or dropping parameters.

Wiring into a real training run requires registering this callback with the
Trainer (a callback that isn't registered never fires) -- see the minimal,
opt-in-only addition in experiments/hicropl_prompt.py, gated behind the
HICROPL_GRADLOG_CSV environment variable so it is a no-op unless explicitly
requested. No training logic (loss, optimizer, forward(), seed, lr, or any
other hyperparameter) is touched by that addition.
"""
import csv
import os

import pytorch_lightning as pl
import torch

GROUPS = ("text_prompt", "visual_prompt", "mapper_lkp", "layernorm")

VISUAL_PROMPT_LEAVES = ("cross_prompts_photo", "cross_prompts_sketch", "ctx_photo", "ctx_sketch")


def _classify(name):
    """name: a key from CustomCLIP.named_parameters(), e.g.
    'text_prompt_photo.ctx', 'visual_visual_learner.cross_prompts_photo.0',
    'visual_visual_learner.photo2sketch_net.linear_q.weight'.

    Returns one of GROUPS, or None for 'clip.*' (handled separately via
    LayerNorm module membership, not name string) or for anything genuinely
    unrecognized (caller must treat None under 'visual_visual_learner.'/
    'text_prompt_*' as a coverage failure -- see _build_groups)."""
    if name.startswith("text_prompt_photo.") or name.startswith("text_prompt_sketch."):
        return "text_prompt"
    if name.startswith("visual_visual_learner."):
        rest = name[len("visual_visual_learner."):]
        leaf = rest.split(".", 1)[0]
        if leaf in VISUAL_PROMPT_LEAVES:
            return "visual_prompt"
        return "mapper_lkp"
    if name.startswith("clip."):
        return None  # classified via LayerNorm module membership, see _build_groups
    return None


def _build_groups(custom_clip):
    """Return {group_name: [nn.Parameter, ...]} for the four groups, and
    raise RuntimeError naming any trainable parameter that doesn't fall into
    one of them (coverage check)."""
    if getattr(custom_clip, "no_prompt_learning", False) or getattr(custom_clip, "use_text_visual_exchange", False):
        raise RuntimeError(
            "GradLogCallback only supports the default architecture "
            "(cfg.no_prompt_learning=False, cfg.use_text_visual_exchange=False) -- "
            f"got no_prompt_learning={getattr(custom_clip, 'no_prompt_learning', None)}, "
            f"use_text_visual_exchange={getattr(custom_clip, 'use_text_visual_exchange', None)}. "
            "Refusing to guess at an unverified grouping for the other architectures; "
            "extend _classify()/_build_groups() in tools/diag_gradlog.py first."
        )

    ln_ids = {
        id(p)
        for m in custom_clip.clip.modules() if isinstance(m, torch.nn.LayerNorm)
        for p in m.parameters()
    }

    groups = {g: [] for g in GROUPS}
    unclassified = []
    for name, p in custom_clip.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("clip."):
            if id(p) in ln_ids:
                groups["layernorm"].append(p)
            else:
                unclassified.append(name)
            continue
        group = _classify(name)
        if group is None:
            unclassified.append(name)
        else:
            groups[group].append(p)

    if unclassified:
        raise RuntimeError(
            "GradLogCallback: the following trainable parameter(s) did not match any "
            f"of the four Q4 groups {GROUPS} -- refusing to silently drop them from "
            f"gradient logging: {unclassified}"
        )
    return groups


class GradLogCallback(pl.Callback):
    def __init__(self, csv_path, every_n_steps=50):
        super().__init__()
        self.csv_path = csv_path
        self.every_n_steps = every_n_steps
        self._groups = None
        self._csv_file = None
        self._writer = None
        self._seen = {g: False for g in GROUPS}
        self._ever_nonzero = {g: False for g in GROUPS}

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return
        self._groups = _build_groups(pl_module.model)
        for g in GROUPS:
            n_params = sum(p.numel() for p in self._groups[g])
            print(f"[diag_gradlog] group={g}: {len(self._groups[g])} tensors, "
                  f"{n_params:,} trainable params")

        write_header = not (os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0)
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)) or ".", exist_ok=True)
        self._csv_file = open(self.csv_path, "a", newline="")
        self._writer = csv.writer(self._csv_file)
        if write_header:
            self._writer.writerow(["step", "epoch", "group", "grad_l2", "param_l2", "ratio"])
            self._csv_file.flush()
        print(f"[diag_gradlog] logging every {self.every_n_steps} optimizer steps to {self.csv_path}")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        step = trainer.global_step + 1
        if step % self.every_n_steps != 0:
            return
        epoch = trainer.current_epoch
        for group_name in GROUPS:
            params = self._groups[group_name]
            if not params:
                continue
            grad_sq = 0.0
            param_sq = 0.0
            for p in params:
                param_sq += p.detach().float().pow(2).sum().item()
                if p.grad is not None:
                    grad_sq += p.grad.detach().float().pow(2).sum().item()
            grad_l2 = grad_sq ** 0.5
            param_l2 = param_sq ** 0.5
            ratio = (grad_l2 / param_l2) if param_l2 > 0 else float("nan")

            self._seen[group_name] = True
            if grad_l2 != 0.0:
                self._ever_nonzero[group_name] = True

            self._writer.writerow([step, epoch, group_name, grad_l2, param_l2, ratio])
        self._csv_file.flush()

    def on_fit_end(self, trainer, pl_module):
        for g in GROUPS:
            if not self._seen[g]:
                print(f"[diag_gradlog] group={g}: never logged (fewer than "
                      f"{self.every_n_steps} optimizer steps ran, or group is empty "
                      f"for this architecture) -- cannot determine dead/alive.")
            elif not self._ever_nonzero[g]:
                print(f"[DEAD] group={g} nhận gradient bằng 0")
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
