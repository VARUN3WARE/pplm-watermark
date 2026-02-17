#!/usr/bin/env python3
"""
Robustness Testing for Watermark System

Tests watermark survival under various text modifications:
- Truncation (removing trailing text)
- Random word deletion
- Word insertion (random common words)
- Homoglyph substitution
- Case perturbation

Each attack is applied at varying intensities to measure
the degradation curve of detection performance.

Usage:
    python robustness_test.py

Output:
    - Console results with per-attack breakdown
    - outputs/robustness_results.png plot

Author: Research Project
Date: February 2026
"""

import torch
import random
import numpy as np
from typing import List, Dict, Tuple

from src.models.pplm import WatermarkGenerator
from src.utils.key_generation import WatermarkKey
from src.watermark.detector import WatermarkDetector
from src.evaluation.visualize import _ensure_output_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Common words for insertion attack
COMMON_WORDS = [
    "the", "a", "an", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "must",
    "very", "quite", "rather", "also", "just", "still", "even",
    "then", "now", "here", "there", "when", "where", "how", "what"
]


def truncate_text(text: str, keep_ratio: float) -> str:
    """Remove trailing portion of text."""
    words = text.split()
    n_keep = max(5, int(len(words) * keep_ratio))
    return " ".join(words[:n_keep])


def delete_random_words(text: str, delete_ratio: float) -> str:
    """Randomly delete words from text."""
    words = text.split()
    if len(words) <= 5:
        return text
    n_delete = int(len(words) * delete_ratio)
    indices_to_delete = set(random.sample(range(len(words)), min(n_delete, len(words) - 5)))
    return " ".join(w for i, w in enumerate(words) if i not in indices_to_delete)


def insert_random_words(text: str, insert_ratio: float) -> str:
    """Insert random common words at random positions."""
    words = text.split()
    n_insert = int(len(words) * insert_ratio)
    for _ in range(n_insert):
        pos = random.randint(0, len(words))
        words.insert(pos, random.choice(COMMON_WORDS))
    return " ".join(words)


def swap_random_words(text: str, swap_ratio: float) -> str:
    """Swap adjacent word pairs randomly."""
    words = text.split()
    n_swaps = int(len(words) * swap_ratio / 2)
    for _ in range(n_swaps):
        if len(words) < 2:
            break
        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def perturb_case(text: str, perturb_ratio: float) -> str:
    """Randomly change case of characters."""
    chars = list(text)
    n_perturb = int(len(chars) * perturb_ratio)
    indices = random.sample(range(len(chars)), min(n_perturb, len(chars)))
    for i in indices:
        if chars[i].isupper():
            chars[i] = chars[i].lower()
        elif chars[i].islower():
            chars[i] = chars[i].upper()
    return "".join(chars)


# Attack registry
ATTACKS = {
    "truncation": {
        "fn": truncate_text,
        "param_name": "keep_ratio",
        "levels": [0.9, 0.75, 0.5, 0.3],
        "labels": ["10%", "25%", "50%", "70%"],
        "description": "Remove trailing text"
    },
    "word_deletion": {
        "fn": delete_random_words,
        "param_name": "delete_ratio",
        "levels": [0.05, 0.1, 0.2, 0.3],
        "labels": ["5%", "10%", "20%", "30%"],
        "description": "Delete random words"
    },
    "word_insertion": {
        "fn": insert_random_words,
        "param_name": "insert_ratio",
        "levels": [0.05, 0.1, 0.2, 0.3],
        "labels": ["5%", "10%", "20%", "30%"],
        "description": "Insert random words"
    },
    "word_swap": {
        "fn": swap_random_words,
        "param_name": "swap_ratio",
        "levels": [0.05, 0.1, 0.2, 0.3],
        "labels": ["5%", "10%", "20%", "30%"],
        "description": "Swap adjacent words"
    },
    "case_perturbation": {
        "fn": perturb_case,
        "param_name": "perturb_ratio",
        "levels": [0.05, 0.1, 0.2, 0.3],
        "labels": ["5%", "10%", "20%", "30%"],
        "description": "Random case changes"
    }
}


# Prompts for testing
TEST_PROMPTS = [
    "The future of artificial intelligence is",
    "Climate change poses significant challenges to",
    "In the world of quantum computing,",
    "The history of ancient civilizations reveals",
    "Modern medicine has made tremendous progress in",
    "The exploration of space continues to",
    "Economic policy affects millions of people through",
    "The development of renewable energy sources",
]


def print_header(text, width=80):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_subheader(text):
    print(f"\n{'~' * 60}")
    print(f"  {text}")
    print(f"{'~' * 60}")


def run_attack_test(
    attack_name: str,
    attack_config: dict,
    watermarked_texts: List[str],
    detector: WatermarkDetector,
    threshold: float = 2.0
) -> Dict:
    """
    Run a single attack at all intensity levels.

    Returns:
        Dict with detection rates and z-scores per level
    """
    results = {"levels": [], "detection_rates": [], "avg_z_scores": []}
    
    for level_idx, level in enumerate(attack_config["levels"]):
        detected_count = 0
        z_scores = []
        
        for text in watermarked_texts:
            # Apply attack
            kwargs = {attack_config["param_name"]: level}
            modified_text = attack_config["fn"](text, **kwargs)
            
            # Detect
            is_detected, score, metadata = detector.detect(
                modified_text, threshold=threshold
            )
            detected_count += int(is_detected)
            z_scores.append(metadata['token']['z_score'])
        
        detection_rate = detected_count / len(watermarked_texts) * 100
        avg_z = np.mean(z_scores)
        
        results["levels"].append(attack_config["labels"][level_idx])
        results["detection_rates"].append(detection_rate)
        results["avg_z_scores"].append(avg_z)
    
    return results


def plot_robustness_results(all_results: Dict[str, Dict], filename: str = "robustness_results.png"):
    """Plot robustness results for all attacks."""
    _ensure_output_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#FF5722', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    
    for idx, (attack_name, results) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        x_positions = range(len(results["levels"]))
        
        ax1.plot(x_positions, results["detection_rates"], 'o-',
                color=color, linewidth=2, markersize=6, label=attack_name)
        
        ax2.plot(x_positions, results["avg_z_scores"], 's-',
                color=color, linewidth=2, markersize=6, label=attack_name)
    
    # Detection rate plot
    ax1.set_xlabel('Attack Intensity')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Watermark Survival Under Attack')
    ax1.set_ylim(-5, 105)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='50% baseline')
    ax1.legend(loc='lower left', fontsize=8)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(['Low', 'Medium', 'High', 'Extreme'])
    
    # Z-score plot
    ax2.set_xlabel('Attack Intensity')
    ax2.set_ylabel('Average Z-Score')
    ax2.set_title('Z-Score Degradation Under Attack')
    ax2.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(['Low', 'Medium', 'High', 'Extreme'])
    
    plt.tight_layout()
    filepath = f"outputs/{filename}"
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    return filepath


def main():
    print_header("WATERMARK ROBUSTNESS TESTING")
    
    # Configuration
    SECRET_KEY = "robustness-test-2026"
    MAX_LENGTH = 100
    THRESHOLD = 2.0
    
    print(f"\nConfiguration:")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"  Max length: {MAX_LENGTH} tokens")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Attacks: {len(ATTACKS)}")
    
    # Initialize system
    print("\nInitializing watermark system...")
    generator = WatermarkGenerator(
        model_name="gpt2",
        secret_key=SECRET_KEY,
        context_dependent=True
    )
    key_manager = WatermarkKey(SECRET_KEY, generator.tokenizer.vocab_size, 768)
    detector = WatermarkDetector(key_manager, generator.tokenizer, context_dependent=True)
    
    print(f"  Device: {generator.device}")
    print(f"  Mode: Context-dependent (v2)")
    
    # Generate watermarked texts
    print_header("GENERATING WATERMARKED TEXTS")
    watermarked_texts = []
    
    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"  [{idx}/{len(TEST_PROMPTS)}] {prompt[:50]}...")
        text = generator.generate(
            prompt=prompt,
            max_length=MAX_LENGTH,
            step_size=0.5,
            kl_lambda=0.0,
            burst_interval=10,
            burst_length=15
        )
        watermarked_texts.append(text)
    
    # Baseline detection (no attack)
    print_header("BASELINE DETECTION (NO ATTACK)")
    baseline_detected = 0
    baseline_z_scores = []
    
    for text in watermarked_texts:
        is_detected, score, metadata = detector.detect(text, threshold=THRESHOLD)
        baseline_detected += int(is_detected)
        baseline_z_scores.append(metadata['token']['z_score'])
    
    baseline_tpr = baseline_detected / len(watermarked_texts) * 100
    baseline_avg_z = np.mean(baseline_z_scores)
    
    print(f"  Detection rate: {baseline_tpr:.0f}% ({baseline_detected}/{len(watermarked_texts)})")
    print(f"  Average z-score: {baseline_avg_z:.2f}")
    
    # Run all attacks
    print_header("RUNNING ATTACK TESTS")
    all_results = {}
    
    for attack_name, attack_config in ATTACKS.items():
        print_subheader(f"Attack: {attack_name} ({attack_config['description']})")
        
        results = run_attack_test(
            attack_name, attack_config,
            watermarked_texts, detector, THRESHOLD
        )
        all_results[attack_name] = results
        
        print(f"\n  {'Level':<12} {'Detection':<15} {'Avg Z-Score':<12}")
        print(f"  {'─' * 40}")
        for i in range(len(results["levels"])):
            print(f"  {results['levels'][i]:<12} "
                  f"{results['detection_rates'][i]:>5.0f}%{'':>9} "
                  f"{results['avg_z_scores'][i]:>8.2f}")
    
    # Generate plot
    print_header("GENERATING PLOTS")
    plot_robustness_results(all_results)
    
    # Summary
    print_header("ROBUSTNESS SUMMARY")
    
    print(f"\n  {'Attack':<20} {'Baseline':<10} {'Low':<10} {'Med':<10} {'High':<10} {'Extreme':<10}")
    print(f"  {'─' * 70}")
    
    for attack_name, results in all_results.items():
        rates = results["detection_rates"]
        print(f"  {attack_name:<20} {baseline_tpr:>5.0f}%    "
              f"{rates[0]:>5.0f}%    {rates[1]:>5.0f}%    "
              f"{rates[2]:>5.0f}%    {rates[3]:>5.0f}%")
    
    # Overall assessment
    print(f"\n  Baseline TPR: {baseline_tpr:.0f}%")
    print(f"  Baseline avg z-score: {baseline_avg_z:.2f}")
    
    # Check which attacks degrade performance significantly
    weak_attacks = []
    for attack_name, results in all_results.items():
        if results["detection_rates"][-1] < baseline_tpr * 0.5:
            weak_attacks.append(attack_name)
    
    if weak_attacks:
        print(f"\n  Vulnerable to: {', '.join(weak_attacks)}")
    else:
        print(f"\n  Robust against all tested attacks")
    
    print("\n" + "=" * 80)
    print("  Robustness testing complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
