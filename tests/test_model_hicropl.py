"""
Unit tests for HiCroPL-SBIR Architecture (CustomCLIP) based on specifications.
"""

import sys
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Mock out src.clip before importing CustomCLIP
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

# Mock out pytorch_lightning before importing CustomCLIP
class MockPL:
    class LightningModule(nn.Module):
        pass
sys.modules['pytorch_lightning'] = MockPL()

# Mock out torchmetrics
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

# Mock out src.utils
class MockUtils:
    @staticmethod
    def freeze_all_but_bn(m):
        pass
sys.modules['src.utils'] = MockUtils()

from src.model_hicropl import CustomCLIP

class DummyCfgTrainer:
    class TrainerConfig:
        N_CTX = 4
        PROMPT_DEPTH = 3
        CROSS_LAYER = 2
        CTX_INIT = "a photo of a"
    COPROMPT = TrainerConfig()
    HICROPL = TrainerConfig()

class DummyCfg:
    TRAINER = DummyCfgTrainer()

class MockTokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(49408, 512))
        
    def forward(self, x):
        return torch.randn(x.shape[0], x.shape[1], 512)

class MockTransformer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        # 3 layers matching PROMPT_DEPTH
        self.resblocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        class MockConv:
            weight = torch.randn(768, 3, 32, 32)
            def __call__(self, x):
                return torch.randn(x.shape[0], 768, 7, 7) # B, width, grid, grid
        self.conv1 = MockConv()
        self.class_embedding = nn.Parameter(torch.randn(768))
        self.positional_embedding = nn.Parameter(torch.randn(50, 768)) # grid^2 + 1
        self.ln_pre = nn.LayerNorm(768)
        self.transformer = MockTransformer(dim=768)
        self.ln_post = nn.LayerNorm(768)
        self.proj = nn.Parameter(torch.randn(768, 512))

    def forward(self, x):
        # Only used by frozen reference encoder, must return [B, 512]
        return torch.randn(x.shape[0], 512)

class MockCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.logit_scale = nn.Parameter(torch.tensor(2.6592)) # ln(1/0.07)
        self.token_embedding = MockTokenEmbedding()
        self.visual = MockViT()
        self.transformer = MockTransformer()
        self.positional_embedding = nn.Parameter(torch.randn(77, 512))
        self.ln_final = nn.LayerNorm(512)
        self.text_projection = nn.Parameter(torch.randn(512, 512))

class TestCustomCLIPSpecification(unittest.TestCase):
    def setUp(self):
        self.cfg = DummyCfg()
        self.clip_model = MockCLIP()
        self.clip_frozen = MockCLIP()

        # Dummy inputs for testing
        self.batch_size = 2
        self.classnames = ["cat", "dog", "car"] # 3 classes

        # Initialize the model being tested with classnames
        self.model = CustomCLIP(self.cfg, self.clip_model, self.clip_frozen, classnames=self.classnames)
        self.photo_img = torch.randn(self.batch_size, 3, 224, 224)
        self.sk_img = torch.randn(self.batch_size, 3, 224, 224)
        self.neg_img = torch.randn(self.batch_size, 3, 224, 224)
        
        self.x = [
            self.sk_img, self.photo_img,
            self.neg_img,
            torch.randint(0, 3, (self.batch_size,)) # labels
        ]

    def test_forward_output_signature(self):
        """
        Specification:
            forward(x, classnames) should return 8 outputs:
            (photo_feat, frozen_photo_feat, logits_photo,
             sketch_feat, frozen_sketch_feat, logits_sketch,
             neg_feat, label)
        """
        output = self.model(self.x, self.classnames)
        
        self.assertEqual(len(output), 8, "Model backward signature does not match expected size of 8")

        # Check types
        for i, item in enumerate(output):
            self.assertIsInstance(item, torch.Tensor, f"Output item {i} is not a Tensor")

    def test_dimensions(self):
        """
        Specification:
            Photo/Sketch/Neg features should be [Batch, 512]
            Frozen references should be [Batch, 512]
            Logits should be [Batch, N_Classes]
        """
        (
            photo_feat, frozen_photo_feat, logits_photo,
            sketch_feat, frozen_sketch_feat, logits_sketch,
            neg_feat, label
        ) = self.model(self.x, self.classnames)
        
        B = self.batch_size
        D = 512
        N_cls = len(self.classnames)
        
        self.assertEqual(photo_feat.shape, (B, D))
        self.assertEqual(frozen_photo_feat.shape, (B, D))
        self.assertEqual(logits_photo.shape, (B, N_cls))
        
        self.assertEqual(sketch_feat.shape, (B, D))
        self.assertEqual(frozen_sketch_feat.shape, (B, D))
        self.assertEqual(logits_sketch.shape, (B, N_cls))
        
        self.assertEqual(neg_feat.shape, (B, D))
        
        self.assertEqual(label.shape, (B,))

    def test_normality(self):
        """
        Specification:
            All features returned from forward (photo_feat, frozen_photo, sketch_feat, frozen_sk, neg_feat)
            MUST be L2 normalized across the embedding dimension to compute valid triplet margins and logits.
        """
        (
            photo_feat, frozen_photo_feat, _,
            sketch_feat, frozen_sketch_feat, _,
            neg_feat, _
        ) = self.model(self.x, self.classnames)

        # Check norms are approximately 1.0
        self.assertTrue(torch.allclose(torch.norm(photo_feat, p=2, dim=-1), torch.ones(self.batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(torch.norm(frozen_photo_feat, p=2, dim=-1), torch.ones(self.batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(torch.norm(sketch_feat, p=2, dim=-1), torch.ones(self.batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(torch.norm(frozen_sketch_feat, p=2, dim=-1), torch.ones(self.batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(torch.norm(neg_feat, p=2, dim=-1), torch.ones(self.batch_size), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
