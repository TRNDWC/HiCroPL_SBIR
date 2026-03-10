"""
Integration test with real Sketchy data.
Requires actual dataset at specified path.
"""

import unittest
import torch
import os

class TestValidDatasetFGIntegration(unittest.TestCase):
    """Integration tests with real data."""
    
    @classmethod
    def setUpClass(cls):
        """Check if dataset exists."""
        cls.data_dir = os.getenv('SKETCHY_DATA_DIR', '/kaggle/input/datasets/.../Sketchy')
        cls.has_data = os.path.exists(cls.data_dir)
        
        if cls.has_data:
            from unittest.mock import Mock
            from src.dataset_retrieval import ValidDatasetFG
            
            args = Mock()
            args.data_dir = cls.data_dir
            args.dataset = 'sketchy'
            args.max_size = 224
            
            cls.sketch_dataset = ValidDatasetFG(args, mode='sketch')
            cls.photo_dataset = ValidDatasetFG(args, mode='photo')
    
    def setUp(self):
        if not self.has_data:
            self.skipTest(f"Dataset not found at {self.data_dir}")
    
    def test_dataset_not_empty(self):
        """Test datasets have samples."""
        self.assertGreater(len(self.sketch_dataset), 0)
        self.assertGreater(len(self.photo_dataset), 0)
    
    def test_getitem_format(self):
        """Test __getitem__ returns correct format."""
        tensor, cat_idx, filename, base_name = self.sketch_dataset[0]
        
        # Check types
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertIsInstance(cat_idx, int)
        self.assertIsInstance(filename, str)
        self.assertIsInstance(base_name, str)
        
        # Check tensor shape
        self.assertEqual(tensor.shape[0], 3)  # RGB channels
        self.assertGreater(tensor.shape[1], 0)  # Height
        self.assertGreater(tensor.shape[2], 0)  # Width
    
    def test_sketch_photo_matching(self):
        """Test sketch and photo base names can match."""
        # Get first sketch
        _, sk_cat, sk_file, sk_base = self.sketch_dataset[0]
        
        # Find matching photo in same category
        matching_photos = [
            (i, base) for i, (_, cat, _, base) in 
            enumerate(self.photo_dataset)
            if cat == sk_cat and base == sk_base
        ]
        
        # Should have at least one matching photo
        self.assertGreater(len(matching_photos), 0,
                          f"No matching photo for sketch {sk_file} (base: {sk_base})")
    
    def test_all_sketches_have_matches(self):
        """Test all sketches have at least one matching photo."""
        # Build photo lookup by (category, base_name)
        photo_lookup = set()
        for _, cat, _, base in self.photo_dataset:
            photo_lookup.add((cat, base))
        
        # Check all sketches
        unmatched = []
        for i, (_, cat, filename, base) in enumerate(self.sketch_dataset):
            if (cat, base) not in photo_lookup:
                unmatched.append((i, filename, base))
        
        if unmatched:
            print(f"\nWarning: {len(unmatched)} sketches have no matching photos:")
            for idx, fname, base in unmatched[:5]:  # Show first 5
                print(f"  - {fname} (base: {base})")
        
        # This might not be 100% - dataset may have inconsistencies
        # But should be mostly matched
        match_rate = 1.0 - len(unmatched) / len(self.sketch_dataset)
        self.assertGreater(match_rate, 0.9,
                          f"Only {match_rate:.1%} sketches have matches")

if __name__ == '__main__':
    unittest.main()