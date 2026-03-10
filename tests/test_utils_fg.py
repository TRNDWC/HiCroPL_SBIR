import unittest
import torch
import os
from src_fg.utils_fg import (
    parse_sketchy_fg_sketch,
    parse_sketchy_fg_photo,
    compute_rank_based_accuracy
)

class TestParseSketchy(unittest.TestCase):
    """Test parsing Sketchy fine-grained filenames."""
    
    def test_sketch_basic(self):
        """Basic sketch: n02691156_10151-1.png → n02691156_10151"""
        result = parse_sketchy_fg_sketch("n02691156_10151-1.png")
        self.assertEqual(result, "n02691156_10151")
    
    def test_sketch_multi_digit_number(self):
        """Multi-digit sketch number: basename-13.png"""
        result = parse_sketchy_fg_sketch("n02691156_10151-13.png")
        self.assertEqual(result, "n02691156_10151")
    
    def test_sketch_with_full_path(self):
        """Full path: /path/to/sketch.png"""
        path = "/kaggle/input/sketchy/sketch/airplane/n02691156_10151-5.png"
        result = parse_sketchy_fg_sketch(path)
        self.assertEqual(result, "n02691156_10151")
    
    def test_sketch_multi_dash_basename(self):
        """Base name with dashes: hot-dog-cat-001-5.png → hot-dog-cat-001"""
        result = parse_sketchy_fg_sketch("hot-dog-cat-001-5.png")
        self.assertEqual(result, "hot-dog-cat-001")
    
    def test_sketch_empty_raises(self):
        """Empty filepath should raise ValueError"""
        with self.assertRaises(ValueError):
            parse_sketchy_fg_sketch("")
    
    def test_sketch_no_dash_raises(self):
        """No dash (invalid format) should raise ValueError"""
        with self.assertRaises(ValueError):
            parse_sketchy_fg_sketch("sketch.png")
    
    def test_sketch_only_one_part_raises(self):
        """Only one part (no sketch number) should raise ValueError"""
        with self.assertRaises(ValueError):
            parse_sketchy_fg_sketch("basename.png")
    
    def test_photo_basic(self):
        """Basic photo: n02691156_10151.jpg → n02691156_10151"""
        result = parse_sketchy_fg_photo("n02691156_10151.jpg")
        self.assertEqual(result, "n02691156_10151")
    
    def test_photo_png_extension(self):
        """Photo with .png: ext_1.png → ext_1"""
        result = parse_sketchy_fg_photo("ext_1.png")
        self.assertEqual(result, "ext_1")
    
    def test_photo_jpeg_extension(self):
        """Photo with .JPEG: image.JPEG → image"""
        result = parse_sketchy_fg_photo("image_001.JPEG")
        self.assertEqual(result, "image_001")
    
    def test_photo_with_full_path(self):
        """Full path photo"""
        path = "/kaggle/input/sketchy/photo/airplane/n02691156_10151.jpg"
        result = parse_sketchy_fg_photo(path)
        self.assertEqual(result, "n02691156_10151")
    
    def test_photo_empty_raises(self):
        """Empty filepath should raise ValueError"""
        with self.assertRaises(ValueError):
            parse_sketchy_fg_photo("")
    
    def test_matching_pair(self):
        """Sketch and photo should extract same base name"""
        sketch_base = parse_sketchy_fg_sketch("n02691156_10151-1.png")
        photo_base = parse_sketchy_fg_photo("n02691156_10151.jpg")
        self.assertEqual(sketch_base, photo_base)
    
    def test_multiple_sketches_one_photo(self):
        """Multiple sketches match to same photo"""
        photo_base = parse_sketchy_fg_photo("n02691156_10151.jpg")
        
        sketch1_base = parse_sketchy_fg_sketch("n02691156_10151-1.png")
        sketch2_base = parse_sketchy_fg_sketch("n02691156_10151-2.png")
        sketch3_base = parse_sketchy_fg_sketch("n02691156_10151-5.png")
        
        self.assertEqual(photo_base, sketch1_base)
        self.assertEqual(photo_base, sketch2_base)
        self.assertEqual(photo_base, sketch3_base)


class TestComputeRankBasedAccuracy(unittest.TestCase):
    """Test rank-based accuracy computation."""
    
    def test_perfect_retrieval(self):
        """All ranks = 1 → 100% accuracy at all k"""
        ranks = torch.tensor([1, 1, 1, 1])
        result = compute_rank_based_accuracy(ranks, [1, 5, 10])
        self.assertEqual(result['acc@1'], 1.0)
        self.assertEqual(result['acc@5'], 1.0)
        self.assertEqual(result['acc@10'], 1.0)
    
    def test_mixed_ranks(self):
        """Mixed ranks → different accuracies"""
        ranks = torch.tensor([2, 1, 5, 1, 10])  # 5 queries
        result = compute_rank_based_accuracy(ranks, [1, 5, 10])
        
        # acc@1: ranks <= 1 → 2/5 = 0.4
        self.assertAlmostEqual(result['acc@1'], 0.4, places=5)
        
        # acc@5: ranks <= 5 → 4/5 = 0.8
        self.assertAlmostEqual(result['acc@5'], 0.8, places=5)
        
        # acc@10: ranks <= 10 → 5/5 = 1.0
        self.assertAlmostEqual(result['acc@10'], 1.0, places=5)
    
    def test_all_fail(self):
        """All ranks > k → 0% accuracy"""
        ranks = torch.tensor([20, 15, 30, 50])
        result = compute_rank_based_accuracy(ranks, [1, 5, 10])
        self.assertEqual(result['acc@1'], 0.0)
        self.assertEqual(result['acc@5'], 0.0)
        self.assertEqual(result['acc@10'], 0.0)
    
    def test_empty_ranks(self):
        """Empty ranks → 0% accuracy"""
        ranks = torch.tensor([])
        result = compute_rank_based_accuracy(ranks, [1, 5])
        self.assertEqual(result['acc@1'], 0.0)
        self.assertEqual(result['acc@5'], 0.0)
    
    def test_custom_k_values(self):
        """Custom k values work correctly"""
        ranks = torch.tensor([1, 2, 3])
        result = compute_rank_based_accuracy(ranks, [1, 2, 3])
        
        # acc@1: 1/3
        self.assertAlmostEqual(result['acc@1'], 1/3, places=5)
        
        # acc@2: 2/3
        self.assertAlmostEqual(result['acc@2'], 2/3, places=5)
        
        # acc@3: 3/3 = 1.0
        self.assertAlmostEqual(result['acc@3'], 1.0, places=5)
    
    def test_boundary_ranks(self):
        """Ranks exactly at boundary"""
        ranks = torch.tensor([1, 5, 10])
        result = compute_rank_based_accuracy(ranks, [1, 5, 10])
        
        # acc@1: rank<=1 → 1/3
        self.assertAlmostEqual(result['acc@1'], 1/3, places=5)
        
        # acc@5: rank<=5 → 2/3
        self.assertAlmostEqual(result['acc@5'], 2/3, places=5)
        
        # acc@10: rank<=10 → 3/3 = 1.0
        self.assertAlmostEqual(result['acc@10'], 1.0, places=5)


if __name__ == '__main__':
    unittest.main()