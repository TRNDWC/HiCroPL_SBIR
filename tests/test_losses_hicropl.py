"""
Unit tests for HiCroPL-SBIR Loss functions based on exact specifications.
"""

import unittest
import torch
import torch.nn.functional as F
from src.losses_hicropl import loss_fn_hicropl

class DummyArgs:
    def __init__(self, lambda_ce=1.0, lambda_consistency=1.0, lambda_text_consistency=1.0, lambda_cross_modal=1.0, cross_modal_loss='triplet', triplet_margin=0.3):
        self.lambda_ce = lambda_ce
        self.lambda_consistency = lambda_consistency
        self.lambda_text_consistency = lambda_text_consistency
        self.lambda_cross_modal = lambda_cross_modal
        self.cross_modal_loss = cross_modal_loss
        self.triplet_margin = triplet_margin

class TestHiCroPLLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.dim = 64
        self.num_classes = 10

        # Create dummy features
        self.photo_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)
        self.sketch_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)
        self.neg_feat = F.normalize(torch.randn(self.batch_size, self.dim), dim=-1)

        # Create dummy text features (shape: [num_classes, dim])
        self.text_feat_photo = F.normalize(torch.randn(self.num_classes, self.dim), dim=-1)
        self.text_feat_sketch = F.normalize(torch.randn(self.num_classes, self.dim), dim=-1)
        
        self.W_photo_desc = F.normalize(torch.randn(self.num_classes, self.dim), dim=-1)
        self.W_sketch_desc = F.normalize(torch.randn(self.num_classes, self.dim), dim=-1)

        # Create dummy logits and labels
        self.logits_photo = torch.randn(self.batch_size, self.num_classes)
        self.logits_sketch = torch.randn(self.batch_size, self.num_classes)
        self.label = torch.randint(0, self.num_classes, (self.batch_size,))

        # Update features tuple to match new unpacking of 10 elements
        self.features = (
            self.photo_feat, self.logits_photo,
            self.sketch_feat, self.logits_sketch,
            self.neg_feat, self.label,
            self.text_feat_photo, self.text_feat_sketch,
            self.W_photo_desc, self.W_sketch_desc
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
        expected_align = self.args.lambda_cross_modal * expected_triplet

        # Specification 2: Classification Cross-Entropy Loss
        expected_ce_photo = F.cross_entropy(self.logits_photo, self.label)
        expected_ce_sketch = F.cross_entropy(self.logits_sketch, self.label)
        expected_ce = self.args.lambda_ce * (expected_ce_photo + expected_ce_sketch)

        # Specification 3: Text Consistency Loss (prompted text vs W_desc)
        expected_cons_text_sketch = (1.0 - F.cosine_similarity(self.text_feat_sketch, self.W_sketch_desc, dim=-1)).mean()
        expected_cons_text_photo = (1.0 - F.cosine_similarity(self.text_feat_photo, self.W_photo_desc, dim=-1)).mean()
        expected_cons_text = self.args.lambda_text_consistency * (expected_cons_text_sketch + expected_cons_text_photo)

        # Specification 4: Visual Consistency Loss (visual feats vs W_desc[label])
        expected_cons_vis_sketch = (1.0 - F.cosine_similarity(self.sketch_feat, self.W_sketch_desc[self.label], dim=-1)).mean()
        expected_cons_vis_photo = (1.0 - F.cosine_similarity(self.photo_feat, self.W_photo_desc[self.label], dim=-1)).mean()
        expected_cons_visual = self.args.lambda_consistency * (expected_cons_vis_sketch + expected_cons_vis_photo)

        # Total expected logic
        expected_total_loss = expected_align + expected_ce + expected_cons_text + expected_cons_visual

        # Model output (Now returns a tuple: total_loss, loss_dict)
        actual_total_loss, loss_dict = loss_fn_hicropl(self.args, self.features)

        self.assertTrue(
            torch.allclose(actual_total_loss, expected_total_loss, atol=1e-5),
            f"Loss fn mismatch. Expected {expected_total_loss}, got {actual_total_loss}"
        )
        self.assertTrue(
            torch.allclose(loss_dict['loss_align'], expected_align, atol=1e-5),
            "loss_align component mismatch in dictionary."
        )

    def test_lambda_scaling(self):
        """
        Verify that lambdas correctly scale their respective losses
        without affecting the triplet alignment loss component.
        """
        args_zeroed = DummyArgs(lambda_ce=0.0, lambda_consistency=0.0, lambda_text_consistency=0.0)
        
        loss_zeroed, loss_dict = loss_fn_hicropl(args_zeroed, self.features)
        
        # When all other lambdas are 0, only triplet alignment loss should remain
        dist_pos = 1.0 - F.cosine_similarity(self.sketch_feat, self.photo_feat)
        dist_neg = 1.0 - F.cosine_similarity(self.sketch_feat, self.neg_feat)
        expected_triplet = F.relu(dist_pos - dist_neg + 0.3).mean()
        
        self.assertTrue(
            torch.allclose(loss_zeroed, expected_triplet, atol=1e-5),
            "Lambda arguments do not scale correctly. Triplet loss should be unaffected."
        )
        self.assertTrue(
            torch.allclose(loss_dict['loss_ce'], torch.tensor(0.0)),
            "Cross entropy loss should be 0 when lambda is 0"
        )

if __name__ == "__main__":
    unittest.main()
