import os
import torch

def parse_sketchy_fg_sketch(filepath: str) -> str:
    if not filepath: 
        raise ValueError("File path is empty.")
    
    filename = os.path.basename(filepath)
    base_filename, ext = os.path.splitext(filename)

    parts = base_filename.split('-')

    if len(parts) < 2:
        raise ValueError("Invalid file name format.")
    
    photo_base = '-'.join(parts[:-1])
    return photo_base


def parse_sketchy_fg_photo(filepath: str) -> str:
    if not filepath: 
        raise ValueError("File path is empty.")
    
    filename = os.path.basename(filepath)
    base_name, _ = os.path.splitext(filename)

    return base_name


def compute_rank_based_accuracy(
        ranks: torch.Tensor,
        top_k_list: list = [1, 5, 10]
) -> dict:
    # Handle empty ranks
    if len(ranks) == 0:
        return {f'acc@{k}': 0.0 for k in top_k_list}
    
    if ranks.ndim != 1:
        raise ValueError("Ranks tensor must be 1-dimensional.")
    
    total_samples = ranks.size(0)
    accuracy_dict = {}

    for k in top_k_list:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        correct_at_k = (ranks <= k).sum().item()
        accuracy_dict[f'acc@{k}'] = correct_at_k / total_samples
    
    return accuracy_dict