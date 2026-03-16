"""Unit tests for the refactored HiCroPL CustomCLIP core blocks."""

import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


class MockModule:
    pass


mock_clip_pkg = MockModule()
mock_clip_module = MockModule()


def mock_tokenize(texts):
    if isinstance(texts, str):
        return torch.zeros(1, 77, dtype=torch.long)
    return torch.zeros(len(texts), 77, dtype=torch.long)


mock_clip_module.tokenize = mock_tokenize
mock_clip_pkg.clip = mock_clip_module

sys.modules['src.clip'] = mock_clip_pkg
sys.modules['src.clip.clip'] = mock_clip_module


class MockPL:
    class LightningModule(nn.Module):
        pass


sys.modules['pytorch_lightning'] = MockPL()


class MockMetrics:
    pass


class MockFunctional:
    pass


class MockRetrieval:
    retrieval_average_precision = lambda *args, **kwargs: 0.0
    retrieval_precision = lambda *args, **kwargs: 0.0


mock_metrics = MockMetrics()
mock_functional = MockFunctional()
mock_retrieval = MockRetrieval()
mock_metrics.functional = mock_functional
mock_functional.retrieval = mock_retrieval

sys.modules['torchmetrics'] = mock_metrics
sys.modules['torchmetrics.functional'] = mock_functional
sys.modules['torchmetrics.functional.retrieval'] = mock_retrieval


import src.model_hicropl as model_hicropl_module


class DummyCfg:
    n_ctx = 4
    prompt_depth = 3
    cross_layer = 2
    ctx_init = "a photo of a"
    ctx_init_sketch = "a sketch of a"
    use_adapter = False
    adapter_reduction = 4
    image_adapter_m = 0.5
    text_adapter_m = 0.5


class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 768, kernel_size=1, bias=False)

    def forward(self, image):
        pooled = image.mean(dim=(2, 3))
        return pooled.repeat(1, 171)[:, :512]


class MockCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.text_projection = nn.Parameter(torch.randn(512, 512))
        self.visual = MockViT()
        self.ln_final = nn.LayerNorm(512)


class DummyPromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx=4, prompt_depth=3, cross_layer=2, ctx_init="", use_fp16=False):
        super().__init__()
        self.n_ctx = n_ctx
        self.prompt_depth = prompt_depth

    def forward(self, classnames):
        n_cls = len(classnames)
        text_input = torch.zeros(n_cls, 77, 512)
        tokenized = torch.zeros(n_cls, 77, dtype=torch.long)
        tokenized[:, 1] = 1
        first_visual_prompt = torch.zeros(self.n_ctx, 768)
        deep_text = [torch.zeros(self.n_ctx, 512) for _ in range(self.prompt_depth - 1)]
        deep_visual = [torch.zeros(self.n_ctx, 768) for _ in range(self.prompt_depth - 1)]
        return text_input, tokenized, first_visual_prompt, deep_text, deep_visual


class DummyTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

    def forward(self, prompts, tokenized_prompts, deep_prompts_text):
        n_cls = prompts.shape[0]
        base = torch.linspace(0.1, 1.0, 512, dtype=prompts.dtype, device=prompts.device)
        offsets = torch.arange(n_cls, dtype=prompts.dtype, device=prompts.device).unsqueeze(1)
        return base.unsqueeze(0).repeat(n_cls, 1) + offsets


class DummyVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

    def forward(self, image, first_visual_prompt, deeper_visual_prompts):
        pooled = image.mean(dim=(2, 3))
        return pooled.repeat(1, 171)[:, :512]


class TestCustomCLIPCoreBlocks(unittest.TestCase):
    def setUp(self):
        self.prompt_patcher = patch.object(model_hicropl_module, 'CrossModalPromptLearner', DummyPromptLearner)
        self.text_patcher = patch.object(model_hicropl_module, 'TextEncoder', DummyTextEncoder)
        self.visual_patcher = patch.object(model_hicropl_module, 'VisualEncoder', DummyVisualEncoder)
        self.prompt_patcher.start()
        self.text_patcher.start()
        self.visual_patcher.start()

        self.cfg = DummyCfg()
        self.model = model_hicropl_module.CustomCLIP(self.cfg, MockCLIP(), MockCLIP())

        self.batch_size = 2
        self.classnames = ["cat", "dog", "car"]
        self.batch = [
            torch.randn(self.batch_size, 3, 224, 224),
            torch.randn(self.batch_size, 3, 224, 224),
            torch.randn(self.batch_size, 3, 224, 224),
            torch.randn(self.batch_size, 3, 224, 224),
            torch.randn(self.batch_size, 3, 224, 224),
            torch.randint(0, len(self.classnames), (self.batch_size,)),
        ]

    def tearDown(self):
        self.prompt_patcher.stop()
        self.text_patcher.stop()
        self.visual_patcher.stop()

    def test_forward_returns_branch_dict(self):
        output = self.model(self.batch, self.classnames)

        self.assertEqual(set(output.keys()), {'photo', 'sketch', 'label'})
        self.assertEqual(output['label'].shape, (self.batch_size,))

        for branch_name in ('photo', 'sketch'):
            branch = output[branch_name]
            self.assertEqual(
                set(branch.keys()),
                {'feature', 'augment_feature', 'text_feature', 'logits', 'augment_logits'},
            )

    def test_branch_dimensions_and_normalization(self):
        output = self.model(self.batch, self.classnames)

        for branch_name in ('photo', 'sketch'):
            branch = output[branch_name]
            self.assertEqual(branch['feature'].shape, (self.batch_size, 512))
            self.assertEqual(branch['augment_feature'].shape, (self.batch_size, 512))
            self.assertEqual(branch['text_feature'].shape, (len(self.classnames), 512))
            self.assertEqual(branch['logits'].shape, (self.batch_size, len(self.classnames)))
            self.assertEqual(branch['augment_logits'].shape, (self.batch_size, len(self.classnames)))

            self.assertTrue(
                torch.allclose(
                    torch.norm(branch['feature'], p=2, dim=-1),
                    torch.ones(self.batch_size),
                    atol=1e-5,
                )
            )
            self.assertTrue(
                torch.allclose(
                    torch.norm(branch['augment_feature'], p=2, dim=-1),
                    torch.ones(self.batch_size),
                    atol=1e-5,
                )
            )
            self.assertTrue(
                torch.allclose(
                    torch.norm(branch['text_feature'], p=2, dim=-1),
                    torch.ones(len(self.classnames)),
                    atol=1e-5,
                )
            )


if __name__ == "__main__":
    unittest.main()