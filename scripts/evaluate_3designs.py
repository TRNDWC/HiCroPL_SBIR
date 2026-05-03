"""
Evaluation of 3 S_text design variants based on detailed criteria.

This script reads metrics from struct_text_3designs.py output and provides:
1. Block structure analysis (Design 1)
2. Alignment quality assessment (all designs)
3. Cross-modal consistency check (Design 3)
4. Modality-specific benefits (Design 2)
5. Symmetry assumption validation (Design 3 vs Design 2)
6. Final recommendations
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path


def evaluate_design1(metrics):
    """
    Design 1: 2C×2C matrix with 4 blocks
    Key questions:
    1. Do cross-modal blocks have higher or lower similarity than within-modality?
    2. Is the matrix discriminative (low std suggests flatness)?
    """
    blocks = metrics.get('design1_blocks', {})
    
    report = {
        'design': 'Design 1 (2C×2C Matrix)',
        'findings': {},
        'concerns': [],
        'recommendations': [],
    }
    
    # Extract stats
    sketch_sketch = blocks.get('sketch_sketch_mean', 0)
    photo_photo = blocks.get('photo_photo_mean', 0)
    cross_modal = blocks.get('sketch_photo_mean', 0)
    
    sketch_sketch_std = blocks.get('sketch_sketch_std', 0)
    photo_photo_std = blocks.get('photo_photo_std', 0)
    
    report['findings'] = {
        'within_sketch_mean': sketch_sketch,
        'within_photo_mean': photo_photo,
        'cross_modal_mean': cross_modal,
        'within_sketch_std': sketch_sketch_std,
        'within_photo_std': photo_photo_std,
    }
    
    # Analysis
    if cross_modal > sketch_sketch or cross_modal > photo_photo:
        report['concerns'].append(
            f'CONCERN: Cross-modal similarity ({cross_modal:.4f}) > within-modality '
            f'(sketch={sketch_sketch:.4f}, photo={photo_photo:.4f}). '
            'This suggests text space conflates sketch-photo as same, not as different modalities.'
        )
    else:
        report['findings']['cross_modal_is_lower'] = True
        report['recommendations'].append(
            'GOOD: Cross-modal block has lower similarity than within-modality blocks.'
        )
    
    if sketch_sketch_std < 0.01 or photo_photo_std < 0.01:
        report['concerns'].append(
            f'CONCERN: Low std in blocks ({sketch_sketch_std:.6f}, {photo_photo_std:.6f}). '
            'Matrix may be too flat to be discriminative.'
        )
    
    # Complexity
    report['findings']['complexity'] = 'High: maintains cross-modal structure explicitly'
    report['findings']['interpretability'] = 'Lower: 4 blocks to monitor'
    
    return report


def evaluate_design2(metrics, design2_corr=None):
    """
    Design 2: Two separate C×C matrices (S_text_sketch, S_text_photo)
    Key questions:
    1. Are the two matrices similar or different?
    2. Do sketch-specific and photo-specific templates improve alignment?
    3. Can we enable/disable per modality?
    """
    report = {
        'design': 'Design 2 (Two Separate C×C Matrices)',
        'findings': {},
        'concerns': [],
        'recommendations': [],
    }
    
    corrs = metrics.get('correlations', {}).get('design2', {})
    
    r_sketch = corrs.get('pearson_sketch', 0)
    r_photo = corrs.get('pearson_photo', 0)
    
    report['findings'] = {
        'pearson_vs_sketch_visual': r_sketch,
        'pearson_vs_photo_visual': r_photo,
    }
    
    # Analysis
    if abs(r_sketch - r_photo) > 0.2:
        report['findings']['modality_differences_significant'] = True
        report['recommendations'].append(
            f'GOOD: Modalities differ significantly (sketch={r_sketch:.4f}, photo={r_photo:.4f}). '
            'Two separate matrices justified for capturing modality-specific geometry.'
        )
    else:
        report['concerns'].append(
            f'CONCERN: Modalities are similar (diff={abs(r_sketch-r_photo):.4f}). '
            'Using two matrices may be redundant.'
        )
    
    if r_sketch < 0.1 and r_photo < 0.1:
        report['concerns'].append(
            'CRITICAL: Both correlations are very low. Text-visual gap likely fundamental, '
            'not template-specific.'
        )
    
    if r_photo > r_sketch:
        report['findings']['photo_aligns_better'] = True
        report['recommendations'].append(
            f'OBSERVATION: Photo template aligns better ({r_photo:.4f} > {r_sketch:.4f}). '
            'Consider using photo geometry as primary anchor.'
        )
    
    report['findings']['complexity'] = 'Medium: two matrices to maintain'
    report['findings']['interpretability'] = 'Good: can analyze each modality separately'
    report['findings']['ablation_capability'] = 'Excellent: can enable/disable per modality'
    
    return report


def evaluate_design3(metrics):
    """
    Design 3: One shared C×C matrix (template: "a photo or a sketch of a")
    Key questions:
    1. Are sketch and photo visual spaces similar (symmetry assumption)?
    2. Does cross-modal consistency improve?
    3. Simpler but makes assumptions?
    """
    report = {
        'design': 'Design 3 (Shared C×C Matrix)',
        'findings': {},
        'concerns': [],
        'recommendations': [],
    }
    
    corrs = metrics.get('correlations', {}).get('design3', {})
    r_sketch = corrs.get('pearson_sketch', 0)
    r_photo = corrs.get('pearson_photo', 0)
    
    consistency = metrics.get('design3_cross_modal_consistency', 0)
    
    report['findings'] = {
        'pearson_vs_sketch_visual': r_sketch,
        'pearson_vs_photo_visual': r_photo,
        'cross_modal_consistency': consistency,
    }
    
    # Analysis: Symmetry assumption
    if abs(r_sketch - r_photo) < 0.15:
        report['findings']['symmetry_assumption_valid'] = True
        report['recommendations'].append(
            f'GOOD: Sketch and photo correlations are similar '
            f'({r_sketch:.4f} vs {r_photo:.4f}). '
            'Symmetry assumption holds — one matrix reasonable.'
        )
    else:
        report['concerns'].append(
            f'CONCERN: Sketch and photo correlations differ significantly '
            f'({abs(r_sketch-r_photo):.4f}). Shared matrix may misalign one modality.'
        )
    
    # Cross-modal consistency
    if consistency > 0.5:
        report['findings']['cross_modal_alignment_good'] = True
        report['recommendations'].append(
            f'GOOD: Cross-modal consistency high ({consistency:.4f}). '
            'Sketch and photo of same class are well-aligned.'
        )
    elif consistency < 0.3:
        report['concerns'].append(
            f'CONCERN: Cross-modal consistency low ({consistency:.4f}). '
            'Sketch and photo embeddings may diverge during training.'
        )
    
    report['findings']['complexity'] = 'Low: single matrix'
    report['findings']['interpretability'] = 'Highest: simple and symmetric'
    report['findings']['ablation_capability'] = 'Limited: all-or-nothing'
    
    return report


def compare_designs(metrics):
    """
    Compare designs directly on alignment quality.
    """
    report = {
        'section': 'Direct Comparison',
        'findings': {},
        'winner': None,
    }
    
    corrs = metrics.get('correlations', {})
    
    # Average alignment per design
    design1_avg = None
    design2_avg = None
    design3_avg = None
    
    if 'design1' in corrs:
        d1 = corrs['design1']
        design1_avg = (d1.get('pearson_sketch', 0) + d1.get('pearson_photo', 0)) / 2
    
    if 'design2' in corrs:
        d2 = corrs['design2']
        design2_avg = (d2.get('pearson_sketch', 0) + d2.get('pearson_photo', 0)) / 2
    
    if 'design3' in corrs:
        d3 = corrs['design3']
        design3_avg = (d3.get('pearson_sketch', 0) + d3.get('pearson_photo', 0)) / 2
    
    scores = {}
    if design1_avg is not None:
        scores['Design 1'] = design1_avg
    if design2_avg is not None:
        scores['Design 2'] = design2_avg
    if design3_avg is not None:
        scores['Design 3'] = design3_avg
    
    report['findings']['avg_alignment_scores'] = scores
    
    if scores:
        best = max(scores, key=scores.get)
        report['winner'] = best
        report['findings']['ranking'] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return report


def generate_final_recommendations(design1_eval, design2_eval, design3_eval, comparison):
    """
    Generate final recommendations based on all evaluations.
    """
    recommendations = {
        'primary_recommendation': None,
        'rationale': [],
        'cautions': [],
        'next_steps': [],
    }
    
    # Count concerns
    concerns_per_design = {
        'Design 1': len(design1_eval.get('concerns', [])),
        'Design 2': len(design2_eval.get('concerns', [])),
        'Design 3': len(design3_eval.get('concerns', [])),
    }
    
    winner = comparison.get('winner')
    
    if winner == 'Design 1':
        recommendations['primary_recommendation'] = 'Design 1: 2C×2C Matrix (Explicit Cross-Modal)'
        recommendations['rationale'].append(
            'Maintains explicit cross-modal structure, allowing fine-grained control '
            'of how sketch and photo geometry interacts.'
        )
        recommendations['cautions'].append(
            'Monitor all 4 blocks during training to ensure cross-modal block contributes meaningfully.'
        )
        recommendations['next_steps'].append(
            'Implement block-wise loss logging: L_ss, L_pp, L_sp, L_ps. '
            'If L_sp + L_ps stagnates, consider disabling cross-modal blocks.'
        )
    
    elif winner == 'Design 2':
        recommendations['primary_recommendation'] = 'Design 2: Two Separate Matrices (Modality-Specific)'
        recommendations['rationale'].append(
            'Allows independent optimization per modality. Sketch and photo can '
            'learn their own anchors based on their respective semantics in text space.'
        )
        recommendations['cautions'].append(
            'If correlations still low, investigate whether CLIP text space aligns with visual space at all.'
        )
        recommendations['next_steps'].append(
            'Log L_struct_sketch and L_struct_photo separately. Ablate: train with only photo, '
            'then only sketch, to quantify modality-specific gains.'
        )
    
    elif winner == 'Design 3':
        recommendations['primary_recommendation'] = 'Design 3: Shared Matrix (Simple & Symmetric)'
        recommendations['rationale'].append(
            'Simplest design with fewest assumptions. Cross-modal consistency score '
            'suggests sketch-photo alignment is naturally enforced by shared geometry.'
        )
        recommendations['cautions'].append(
            'Verify that shared matrix does not "average away" important modality differences. '
            'Check if unseen class zero-shot retrieval is unbalanced (sketch query overfits one modality).'
        )
        recommendations['next_steps'].append(
            'Monitor zero-shot mAP separately for sketch-photo and photo-sketch retrieval. '
            'If one direction is much worse, reconsider Design 2.'
        )
    
    else:
        recommendations['primary_recommendation'] = 'Uncertain — Further Investigation Needed'
        recommendations['cautions'].append(
            'No clear winner across designs. This suggests text-visual gap is fundamental; '
            'no design variant fully solves it.'
        )
        recommendations['next_steps'].append(
            'Consider whether structural loss is the right approach. '
            'Alternative: use text geometry for initialization only, then fine-tune visually.'
        )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3 S_text designs')
    parser.add_argument('--metrics-file', default='results/3designs_analysis/metrics.json')
    parser.add_argument('--out-dir', default='results/3designs_analysis')
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics_file):
        print(f'ERROR: Metrics file not found: {args.metrics_file}')
        return
    
    with open(args.metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print(f'\n{"="*70}')
    print(f'EVALUATION: 3 S_text Design Variants')
    print(f'{"="*70}\n')
    
    # Evaluate each design
    design1_eval = evaluate_design1(metrics)
    design2_eval = evaluate_design2(metrics)
    design3_eval = evaluate_design3(metrics)
    comparison = compare_designs(metrics)
    final_recs = generate_final_recommendations(design1_eval, design2_eval, design3_eval, comparison)
    
    # Print Design 1 evaluation
    print(f'\n{design1_eval["design"]}')
    print(f'{"-"*70}')
    print('Findings:')
    for key, val in design1_eval['findings'].items():
        print(f'  {key}: {val}')
    if design1_eval['concerns']:
        print('Concerns:')
        for concern in design1_eval['concerns']:
            print(f'  ⚠️  {concern}')
    if design1_eval['recommendations']:
        print('Recommendations:')
        for rec in design1_eval['recommendations']:
            print(f'  ✓ {rec}')
    
    # Print Design 2 evaluation
    print(f'\n{design2_eval["design"]}')
    print(f'{"-"*70}')
    print('Findings:')
    for key, val in design2_eval['findings'].items():
        print(f'  {key}: {val}')
    if design2_eval['concerns']:
        print('Concerns:')
        for concern in design2_eval['concerns']:
            print(f'  ⚠️  {concern}')
    if design2_eval['recommendations']:
        print('Recommendations:')
        for rec in design2_eval['recommendations']:
            print(f'  ✓ {rec}')
    
    # Print Design 3 evaluation
    print(f'\n{design3_eval["design"]}')
    print(f'{"-"*70}')
    print('Findings:')
    for key, val in design3_eval['findings'].items():
        print(f'  {key}: {val}')
    if design3_eval['concerns']:
        print('Concerns:')
        for concern in design3_eval['concerns']:
            print(f'  ⚠️  {concern}')
    if design3_eval['recommendations']:
        print('Recommendations:')
        for rec in design3_eval['recommendations']:
            print(f'  ✓ {rec}')
    
    # Print comparison
    print(f'\n{comparison["section"]}')
    print(f'{"-"*70}')
    print(f'Average alignment scores:')
    for design, score in comparison['findings'].get('avg_alignment_scores', {}).items():
        print(f'  {design}: {score:.4f}')
    
    if comparison.get('winner'):
        print(f'\nWINNER: {comparison["winner"]}')
        ranking = comparison['findings'].get('ranking', [])
        for i, (design, score) in enumerate(ranking, 1):
            print(f'  {i}. {design}: {score:.4f}')
    
    # Print final recommendations
    print(f'\n{"="*70}')
    print(f'FINAL RECOMMENDATION')
    print(f'{"="*70}')
    print(f'{final_recs["primary_recommendation"]}')
    
    print(f'\nRationale:')
    for rationale in final_recs['rationale']:
        print(f'  • {rationale}')
    
    if final_recs['cautions']:
        print(f'\nCautions:')
        for caution in final_recs['cautions']:
            print(f'  ⚠️  {caution}')
    
    if final_recs['next_steps']:
        print(f'\nNext Steps:')
        for step in final_recs['next_steps']:
            print(f'  → {step}')
    
    print(f'\n{"="*70}\n')
    
    # Save full evaluation report
    full_report = {
        'design1': design1_eval,
        'design2': design2_eval,
        'design3': design3_eval,
        'comparison': comparison,
        'final_recommendations': final_recs,
    }
    
    report_path = os.path.join(args.out_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f'Full report saved to: {report_path}')


if __name__ == '__main__':
    main()
