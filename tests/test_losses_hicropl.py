"""
Unit tests for HiCroPL-SBIR Loss functions based on exact specifications.
"""

import unittest
import torch
import torch.nn.functional as F
from src.losses_hicropl import loss_fn_hicropl

class DummyArgs:
    def __init__(self, lambda_ce=1.0, lambda_consist=0.1):
        self.lambda_ce = lambda_ce
        self.lambda_consist = lambda_consist


class TestHiCroPLLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.dim = 64
        self.num_classes = 10

        # Create dummy features
        self.photo_feat = torch.randn(self.batch_size, self.dim)
        self.photo_feat = F.normalize(self.photo_feat, dim=-1)

        self.frozen_photo_feat = torch.randn(self.batch_size, self.dim)
        self.frozen_photo_feat = F.normalize(self.frozen_photo_feat, dim=-1)

        self.sketch_feat = torch.randn(self.batch_size, self.dim)
        self.sketch_feat = F.normalize(self.sketch_feat, dim=-1)

        self.frozen_sketch_feat = torch.randn(self.batch_size, self.dim)
        self.frozen_sketch_feat = F.normalize(self.frozen_sketch_feat, dim=-1)

        self.neg_feat = torch.randn(self.batch_size, self.dim)
        self.neg_feat = F.normalize(self.neg_feat, dim=-1)

        # Create dummy logits and labels
        self.logits_photo = torch.randn(self.batch_size, self.num_classes)
        self.logits_sketch = torch.randn(self.batch_size, self.num_classes)
        self.label = torch.randint(0, self.num_classes, (self.batch_size,))

        self.features = (
            self.photo_feat, self.frozen_photo_feat, self.logits_photo,
            self.sketch_feat, self.frozen_sketch_feat, self.logits_sketch,
            self.neg_feat, self.label
        )
        self.args = DummyArgs()

    def test_loss_components_match_specification(self):
        """
        Calculates loss perfectly manually from mathematical specs
        to ensure loss_fn_hicropl strictly implements the intended logic.
        """
        # Specification 1: Triplet loss with distance = 1 - cosine
        dist_pos = 1.0 - F.cosine_similarity(self.sketch_feat, self.photo_feat)
        dist_neg = 1.0 - F.cosine_similarity(self.sketch_feat, self.neg_feat)
        expected_triplet = F.relu(dist_pos - dist_neg + 0.3).mean()

        # Specification 2: Classification Cross-Entropy Loss
        expected_ce_photo = F.cross_entropy(self.logits_photo, self.label)
        expected_ce_sketch = F.cross_entropy(self.logits_sketch, self.label)
        expected_cls = self.args.lambda_ce * (expected_ce_photo + expected_ce_sketch)

        # Specification 3: Consistency Loss maintaining distance to frozen models
        expected_consist_photo = (1.0 - F.cosine_similarity(self.photo_feat, self.frozen_photo_feat)).mean()
        expected_consist_sketch = (1.0 - F.cosine_similarity(self.sketch_feat, self.frozen_sketch_feat)).mean()
        expected_consist = self.args.lambda_consist * (expected_consist_photo + expected_consist_sketch)

        # Total expected logic
        expected_total_loss = expected_triplet + expected_cls + expected_consist

        # Model output
        actual_total_loss = loss_fn_hicropl(self.args, self.features)

        self.assertTrue(
            torch.allclose(actual_total_loss, expected_total_loss, atol=1e-5),
            f"Loss fn mismatch. Expected {expected_total_loss}, got {actual_total_loss}"
        )

    def test_lambda_scaling(self):
        """
        Verify that lambda_ce and lambda_consist correctly scale their respective losses
        without affecting the triplet loss component.
        """
        args_zeroed = DummyArgs(lambda_ce=0.0, lambda_consist=0.0)
        
        loss_zeroed = loss_fn_hicropl(args_zeroed, self.features)
        
        # When lambdas are 0, only triplet loss should remain
        dist_pos = 1.0 - F.cosine_similarity(self.sketch_feat, self.photo_feat)
        dist_neg = 1.0 - F.cosine_similarity(self.sketch_feat, self.neg_feat)
        expected_triplet = F.relu(dist_pos - dist_neg + 0.3).mean()
        
        self.assertTrue(
            torch.allclose(loss_zeroed, expected_triplet, atol=1e-5),
            "Lambda arguments do not scale correctly. Triplet loss should be unaffected."
        )

if __name__ == "__main__":
    unittest.main()
