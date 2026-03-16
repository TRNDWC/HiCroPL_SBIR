"""Unit tests for the refactored HiCroPL loss function."""

import unittest

import torch
import torch.nn.functional as F

from src.losses_hicropl import cross_loss, loss_fn_hicropl


class DummyArgs:
    def __init__(
        self,
        temperature=0.07,
        lambda_cross_modal=1.0,
        lambda_ce=1.0,
        lambda_ce_aug=1.0,
        lambda_consistency=1.0,
    ):
        self.temperature = temperature
        self.lambda_cross_modal = lambda_cross_modal
        self.lambda_ce = lambda_ce
        self.lambda_ce_aug = lambda_ce_aug
        self.lambda_consistency = lambda_consistency


class TestHiCroPLLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.dim = 64
        self.num_classes = 10

        self.photo_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)
        self.photo_aug_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)
        self.sketch_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)
        self.sketch_aug_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)

        self.text_photo_feat = F.normalize(torch.randn(self.num_classes, self.dim), dim=-1)
        self.text_sketch_feat = F.normalize(torch.randn(self.num_classes, self.dim), dim=-1)

        self.logits_photo = torch.randn(self.batch_size, self.num_classes)
        self.logits_photo_aug = torch.randn(self.batch_size, self.num_classes)
        self.logits_sketch = torch.randn(self.batch_size, self.num_classes)
        self.logits_sketch_aug = torch.randn(self.batch_size, self.num_classes)
        self.label = torch.randint(0, self.num_classes, (self.batch_size,))

        self.features = {
            'photo': {
                'feature': self.photo_feat,
                'augment_feature': self.photo_aug_feat,
                'text_feature': self.text_photo_feat,
                'logits': self.logits_photo,
                'augment_logits': self.logits_photo_aug,
            },
            'sketch': {
                'feature': self.sketch_feat,
                'augment_feature': self.sketch_aug_feat,
                'text_feature': self.text_sketch_feat,
                'logits': self.logits_sketch,
                'augment_logits': self.logits_sketch_aug,
            },
            'label': self.label,
        }
        self.args = DummyArgs()

    def test_loss_components_match_specification(self):
        expected_cross_modal = self.args.lambda_cross_modal * cross_loss(
            self.sketch_feat,
            self.photo_feat,
            self.args.temperature,
        )
        expected_ce = self.args.lambda_ce * (
            F.cross_entropy(self.logits_photo, self.label)
            + F.cross_entropy(self.logits_sketch, self.label)
        )
        expected_ce_aug = self.args.lambda_ce_aug * (
            F.cross_entropy(self.logits_photo_aug, self.label)
            + F.cross_entropy(self.logits_sketch_aug, self.label)
        )
        expected_consistency = self.args.lambda_consistency * (
            cross_loss(self.photo_feat, self.photo_aug_feat, self.args.temperature)
            + cross_loss(self.sketch_feat, self.sketch_aug_feat, self.args.temperature)
        )
        expected_total_loss = (
            expected_cross_modal
            + expected_ce
            + expected_ce_aug
            + expected_consistency
        )

        actual_total_loss = loss_fn_hicropl(self.args, self.features)

        self.assertTrue(
            torch.allclose(actual_total_loss, expected_total_loss, atol=1e-5),
            f"Loss fn mismatch. Expected {expected_total_loss}, got {actual_total_loss}",
        )

    def test_lambda_scaling(self):
        args_zeroed = DummyArgs(
            lambda_cross_modal=0.0,
            lambda_ce=0.0,
            lambda_ce_aug=0.0,
            lambda_consistency=0.0,
        )

        loss_zeroed = loss_fn_hicropl(args_zeroed, self.features)

        self.assertTrue(
            torch.allclose(loss_zeroed, torch.tensor(0.0), atol=1e-5),
            "All weighted loss terms should be disabled when lambdas are zero.",
        )


if __name__ == "__main__":
    unittest.main()