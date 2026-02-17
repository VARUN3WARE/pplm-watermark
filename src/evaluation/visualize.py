"""
Visualization module for watermark analysis.

Generates publication-quality plots for:
- Detection score distributions (watermarked vs clean)
- ROC curves
- Green token heatmaps
- Quality vs detection trade-off analysis

All plots are saved to the outputs/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple
import os


# Style configuration for clean, professional plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = "outputs"


def _ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_score_distribution(
    watermarked_scores: List[float],
    clean_scores: List[float],
    threshold: float = 2.0,
    title: str = "Detection Score Distribution",
    filename: str = "score_distribution.png"
):
    """
    Plot histogram of z-scores for watermarked vs clean text.

    Args:
        watermarked_scores: Z-scores from watermarked text samples
        clean_scores: Z-scores from clean text samples
        threshold: Detection threshold line
        title: Plot title
        filename: Output filename
    """
    _ensure_output_dir()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine bin range
    all_scores = watermarked_scores + clean_scores
    bin_min = min(all_scores) - 0.5
    bin_max = max(all_scores) + 0.5
    bins = np.linspace(bin_min, bin_max, 25)
    
    ax.hist(clean_scores, bins=bins, alpha=0.6, color='#2196F3',
            label=f'Clean (n={len(clean_scores)})', edgecolor='white', linewidth=0.5)
    ax.hist(watermarked_scores, bins=bins, alpha=0.6, color='#FF5722',
            label=f'Watermarked (n={len(watermarked_scores)})', edgecolor='white', linewidth=0.5)
    
    # Threshold line
    ax.axvline(x=threshold, color='#333333', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold}')
    
    # Add mean markers
    wm_mean = np.mean(watermarked_scores)
    clean_mean = np.mean(clean_scores)
    ax.axvline(x=wm_mean, color='#FF5722', linestyle=':', alpha=0.7)
    ax.axvline(x=clean_mean, color='#2196F3', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Z-Score')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    # Add separation annotation
    separation = wm_mean - clean_mean
    ax.text(0.02, 0.95, f'Separation: {separation:.2f} std dev',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_roc_curve(
    watermarked_scores: List[float],
    clean_scores: List[float],
    title: str = "ROC Curve",
    filename: str = "roc_curve.png"
):
    """
    Plot ROC curve from detection scores.

    Args:
        watermarked_scores: Z-scores from watermarked text (positive class)
        clean_scores: Z-scores from clean text (negative class)
        title: Plot title
        filename: Output filename
    """
    _ensure_output_dir()
    
    # Compute ROC points by sweeping threshold
    all_scores = sorted(set(watermarked_scores + clean_scores))
    thresholds = np.linspace(min(all_scores) - 1, max(all_scores) + 1, 200)
    
    tpr_list = []
    fpr_list = []
    
    for thresh in thresholds:
        tp = sum(1 for s in watermarked_scores if s > thresh)
        fn = sum(1 for s in watermarked_scores if s <= thresh)
        fp = sum(1 for s in clean_scores if s > thresh)
        tn = sum(1 for s in clean_scores if s <= thresh)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Compute AUC using trapezoidal rule
    # Sort by FPR for proper AUC computation
    sorted_pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in sorted_pairs]
    tpr_sorted = [p[1] for p in sorted_pairs]
    auc = np.trapezoid(tpr_sorted, fpr_sorted)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.plot(fpr_list, tpr_list, color='#FF5722', linewidth=2,
            label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random classifier')
    
    # Mark the point at threshold=2.0
    tp_at_2 = sum(1 for s in watermarked_scores if s > 2.0)
    fp_at_2 = sum(1 for s in clean_scores if s > 2.0)
    tpr_at_2 = tp_at_2 / len(watermarked_scores) if watermarked_scores else 0
    fpr_at_2 = fp_at_2 / len(clean_scores) if clean_scores else 0
    ax.plot(fpr_at_2, tpr_at_2, 'ko', markersize=8, label=f'Threshold=2.0')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_green_token_heatmap(
    tokens: List[int],
    per_token_green: List[bool],
    tokenizer,
    title: str = "Green Token Heatmap",
    filename: str = "green_heatmap.png",
    max_display_tokens: int = 60
):
    """
    Visualize which tokens are green/red in a watermarked text.

    Args:
        tokens: List of token IDs
        per_token_green: Boolean list (True=green, False=red)
        tokenizer: Tokenizer for decoding tokens
        title: Plot title
        filename: Output filename
        max_display_tokens: Maximum tokens to display
    """
    _ensure_output_dir()
    
    # Limit to max_display_tokens
    display_tokens = tokens[1:max_display_tokens + 1]  # Skip first (no prev context)
    display_green = per_token_green[:max_display_tokens]
    
    n = len(display_tokens)
    if n == 0:
        return
    
    # Decode tokens to text
    token_texts = [tokenizer.decode([t]).replace('\n', '\\n') for t in display_tokens]
    
    # Create figure with colored cells
    cols_per_row = 15
    n_rows = (n + cols_per_row - 1) // cols_per_row
    
    fig, ax = plt.subplots(figsize=(14, max(2, n_rows * 1.2)))
    
    for i in range(n):
        row = i // cols_per_row
        col = i % cols_per_row
        
        color = '#4CAF50' if display_green[i] else '#f44336'  # green / red
        alpha = 0.6
        
        rect = plt.Rectangle((col, -row), 0.95, 0.85, 
                             facecolor=color, alpha=alpha,
                             edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        
        # Token text (truncate if too long)
        text = token_texts[i][:8]
        ax.text(col + 0.475, -row + 0.425, text,
                ha='center', va='center', fontsize=7,
                fontweight='bold', color='white')
    
    ax.set_xlim(-0.1, cols_per_row + 0.1)
    ax.set_ylim(-n_rows + 0.1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, pad=15)
    
    # Legend
    green_patch = mpatches.Patch(color='#4CAF50', alpha=0.6, label='Green token')
    red_patch = mpatches.Patch(color='#f44336', alpha=0.6, label='Red token')
    green_ratio = sum(display_green) / len(display_green) * 100
    ax.legend(handles=[green_patch, red_patch], loc='upper right',
              title=f'Green ratio: {green_ratio:.0f}%')
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_quality_vs_strength(
    step_sizes: List[float],
    z_scores: List[float],
    ppl_increases: List[float],
    title: str = "Quality vs Detection Trade-off",
    filename: str = "quality_vs_strength.png"
):
    """
    Plot trade-off between detection strength and quality degradation
    across different step sizes.

    Args:
        step_sizes: List of step size values tested
        z_scores: Average z-scores at each step size
        ppl_increases: Perplexity increase (%) at each step size
        title: Plot title
        filename: Output filename
    """
    _ensure_output_dir()
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = '#FF5722'
    color2 = '#2196F3'
    
    ax1.set_xlabel('Step Size')
    ax1.set_ylabel('Average Z-Score', color=color1)
    line1 = ax1.plot(step_sizes, z_scores, 'o-', color=color1, linewidth=2,
                     markersize=6, label='Z-Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='Threshold (2.0)')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Perplexity Increase (%)', color=color2)
    line2 = ax2.plot(step_sizes, ppl_increases, 's-', color=color2, linewidth=2,
                     markersize=6, label='PPL Increase')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title(title)
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def plot_detection_comparison(
    v1_scores: List[float],
    v2_scores: List[float],
    clean_scores: List[float],
    threshold: float = 2.0,
    title: str = "v1 vs v2 Detection Comparison",
    filename: str = "v1_v2_comparison.png"
):
    """
    Compare detection distributions between v1 (fixed) and v2 (context-dependent).

    Args:
        v1_scores: Z-scores from v1 watermarked text
        v2_scores: Z-scores from v2 watermarked text
        clean_scores: Z-scores from clean text
        threshold: Detection threshold
        title: Plot title
        filename: Output filename
    """
    _ensure_output_dir()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    all_scores = v1_scores + v2_scores + clean_scores
    bin_min = min(all_scores) - 0.5
    bin_max = max(all_scores) + 0.5
    bins = np.linspace(bin_min, bin_max, 25)
    
    ax.hist(clean_scores, bins=bins, alpha=0.5, color='#9E9E9E',
            label=f'Clean (n={len(clean_scores)})', edgecolor='white')
    ax.hist(v1_scores, bins=bins, alpha=0.5, color='#FF9800',
            label=f'v1 Fixed (n={len(v1_scores)})', edgecolor='white')
    ax.hist(v2_scores, bins=bins, alpha=0.5, color='#4CAF50',
            label=f'v2 Context (n={len(v2_scores)})', edgecolor='white')
    
    ax.axvline(x=threshold, color='#333333', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold}')
    
    ax.set_xlabel('Z-Score')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    # Add TPR annotations
    v1_tpr = sum(1 for s in v1_scores if s > threshold) / len(v1_scores) * 100 if v1_scores else 0
    v2_tpr = sum(1 for s in v2_scores if s > threshold) / len(v2_scores) * 100 if v2_scores else 0
    
    ax.text(0.02, 0.95, f'v1 TPR: {v1_tpr:.0f}% | v2 TPR: {v2_tpr:.0f}%',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath
