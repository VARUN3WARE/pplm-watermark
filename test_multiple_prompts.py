#!/usr/bin/env python3
"""
Multi-Prompt Watermark Evaluation (v2)

Comprehensive testing of the v2 watermarking system with context-dependent
green lists across diverse prompts. Evaluates:
- Detection consistency across 10 different prompts
- True positive rate and false positive rate
- Signal strength and statistical separation
- Quality impact (perplexity increase)
- Generates ROC curve and score distribution plots

Usage:
    python test_multiple_prompts.py

Author: Research Project
Date: February 2026
"""

import torch
from src.models.pplm import WatermarkGenerator
from src.utils.key_generation import WatermarkKey
from src.watermark.detector import WatermarkDetector
from src.evaluation.quality import QualityEvaluator
from src.evaluation.visualize import (
    plot_score_distribution,
    plot_roc_curve,
    plot_green_token_heatmap
)

# Test prompts covering different topics and styles
TEST_PROMPTS = [
    "The future of artificial intelligence is",
    "Climate change poses significant challenges to",
    "In the world of quantum computing,",
    "The history of ancient civilizations reveals",
    "Modern medicine has made tremendous progress in",
    "The exploration of space continues to",
    "Economic policy affects millions of people through",
    "The development of renewable energy sources",
    "In the field of neuroscience, researchers",
    "The evolution of human language demonstrates"
]

def print_header(text, width=80):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def print_subheader(text):
    print(f"\n{'~' * 60}")
    print(f"  {text}")
    print(f"{'~' * 60}")

def main():
    print_header("MULTI-PROMPT WATERMARK TESTING (v2)")
    
    # Configuration
    SECRET_KEY = "multi-test-secret-2026"
    MAX_LENGTH = 80
    THRESHOLD = 2.0
    
    print(f"\nTest Configuration:")
    print(f"   Number of prompts: {len(TEST_PROMPTS)}")
    print(f"   Max length: {MAX_LENGTH} tokens")
    print(f"   Detection threshold: {THRESHOLD}")
    print(f"   Secret key: {SECRET_KEY}")
    print(f"   Mode: Context-dependent green lists (v2)")
    
    # Initialize system
    print("\nInitializing watermark system...")
    generator = WatermarkGenerator(
        model_name="gpt2",
        secret_key=SECRET_KEY,
        context_dependent=True
    )
    key_manager = WatermarkKey(SECRET_KEY, generator.tokenizer.vocab_size, 768)
    detector = WatermarkDetector(key_manager, generator.tokenizer, context_dependent=True)
    evaluator = QualityEvaluator(model_name="gpt2")
    
    print(f"   Device: {generator.device}")
    print(f"   Model: GPT-2 (124M)")
    
    # Storage for results
    results = []
    all_wm_z_scores = []
    all_clean_z_scores = []
    
    # Test each prompt
    print_header("TESTING INDIVIDUAL PROMPTS")
    
    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print_subheader(f"Test #{idx}: {prompt[:50]}...")
        
        # Generate watermarked text
        print(f"  Generating watermarked text...")
        watermarked_text = generator.generate(
            prompt=prompt,
            max_length=MAX_LENGTH,
            step_size=0.5,
            kl_lambda=0.0,
            burst_interval=10,
            burst_length=15
        )
        
        # Generate clean text for comparison
        print(f"  Generating clean text...")
        clean_text = generator.generate_clean(
            prompt=prompt,
            max_length=MAX_LENGTH
        )
        
        # Detect watermark
        detected_wm, score_wm, metadata_wm = detector.detect(
            watermarked_text,
            threshold=THRESHOLD
        )
        
        detected_clean, score_clean, metadata_clean = detector.detect(
            clean_text,
            threshold=THRESHOLD
        )
        
        # Extract metrics
        token_meta_wm = metadata_wm['token']
        token_meta_clean = metadata_clean['token']
        green_ratio_wm = token_meta_wm['green_ratio']
        z_score_wm = token_meta_wm['z_score']
        green_ratio_clean = token_meta_clean['green_ratio']
        z_score_clean = token_meta_clean['z_score']
        
        all_wm_z_scores.append(z_score_wm)
        all_clean_z_scores.append(z_score_clean)
        
        # Compute quality
        quality_results = evaluator.compute_perplexity_increase(watermarked_text, clean_text)
        ppl_increase = quality_results['percentage_increase']
        
        # Store results
        result = {
            'prompt': prompt,
            'watermarked_detected': detected_wm,
            'clean_detected': detected_clean,
            'wm_green_ratio': green_ratio_wm,
            'clean_green_ratio': green_ratio_clean,
            'wm_z_score': z_score_wm,
            'clean_z_score': z_score_clean,
            'ppl_increase': ppl_increase,
            'watermarked_text': watermarked_text,
            'clean_text': clean_text,
            'wm_metadata': metadata_wm,
        }
        results.append(result)
        
        # Print individual results
        wm_status = "DETECTED" if detected_wm else "NOT DETECTED"
        clean_status = "FALSE POS" if detected_clean else "CLEAN"
        print(f"\n  Results:")
        print(f"   Watermarked: {green_ratio_wm*100:.1f}% green | Z-score: {z_score_wm:.2f} | {wm_status}")
        print(f"   Clean:       {green_ratio_clean*100:.1f}% green | Z-score: {z_score_clean:.2f} | {clean_status}")
        print(f"   Perplexity increase: {ppl_increase:.1f}%")
        print(f"   Separation: {z_score_wm - z_score_clean:.2f} std dev")
        
        # Show text snippets
        print(f"\n  Watermarked: {watermarked_text[:120]}...")
        print(f"  Clean:       {clean_text[:120]}...")
    
    # Generate visualizations
    print_header("GENERATING VISUALIZATIONS")
    
    plot_score_distribution(
        watermarked_scores=all_wm_z_scores,
        clean_scores=all_clean_z_scores,
        threshold=THRESHOLD,
        title="v2 Detection Score Distribution (10 Prompts)",
        filename="multi_prompt_scores.png"
    )
    
    plot_roc_curve(
        watermarked_scores=all_wm_z_scores,
        clean_scores=all_clean_z_scores,
        title="v2 ROC Curve (10 Prompts)",
        filename="multi_prompt_roc.png"
    )
    
    # Generate heatmap for first prompt that was detected
    for r in results:
        if r['watermarked_detected'] and 'per_token_green' in r['wm_metadata']['token']:
            tokens = generator.tokenizer.encode(r['watermarked_text'])
            plot_green_token_heatmap(
                tokens=tokens,
                per_token_green=r['wm_metadata']['token']['per_token_green'],
                tokenizer=generator.tokenizer,
                title=f"Green Token Heatmap: {r['prompt'][:40]}...",
                filename="multi_prompt_heatmap.png"
            )
            break
    
    # Aggregate statistics
    print_header("AGGREGATE STATISTICS")
    
    total_tests = len(results)
    wm_detected = sum(1 for r in results if r['watermarked_detected'])
    false_positives = sum(1 for r in results if r['clean_detected'])
    
    avg_wm_green = sum(r['wm_green_ratio'] for r in results) / total_tests
    avg_clean_green = sum(r['clean_green_ratio'] for r in results) / total_tests
    avg_wm_z = sum(r['wm_z_score'] for r in results) / total_tests
    avg_clean_z = sum(r['clean_z_score'] for r in results) / total_tests
    avg_ppl_increase = sum(r['ppl_increase'] for r in results) / total_tests
    
    avg_separation = avg_wm_z - avg_clean_z
    
    print(f"\nDetection Performance:")
    print(f"   True Positive Rate:  {wm_detected}/{total_tests} ({100*wm_detected/total_tests:.1f}%)")
    print(f"   False Positive Rate: {false_positives}/{total_tests} ({100*false_positives/total_tests:.1f}%)")
    print(f"   Accuracy: {(wm_detected + (total_tests - false_positives))/(2*total_tests)*100:.1f}%")
    
    print(f"\nSignal Strength:")
    print(f"   Avg watermarked green ratio: {avg_wm_green*100:.1f}%")
    print(f"   Avg clean green ratio:       {avg_clean_green*100:.1f}%")
    print(f"   Difference: {(avg_wm_green - avg_clean_green)*100:.1f}%")
    
    print(f"\nStatistical Metrics:")
    print(f"   Avg watermarked Z-score: {avg_wm_z:.2f}")
    print(f"   Avg clean Z-score:       {avg_clean_z:.2f}")
    print(f"   Avg separation:          {avg_separation:.2f} std dev")
    
    print(f"\nQuality Impact:")
    print(f"   Avg perplexity increase: {avg_ppl_increase:.1f}%")
    
    # Detailed breakdown
    print_header("DETAILED BREAKDOWN")
    
    print(f"\n{'#':<3} {'Prompt':<45} {'WM Detect':<11} {'FP':<6} {'Z-score':<8} {'PPL inc':<8}")
    print("~" * 90)
    
    for idx, r in enumerate(results, 1):
        prompt_short = r['prompt'][:42] + "..." if len(r['prompt']) > 45 else r['prompt']
        wm_status = "PASS" if r['watermarked_detected'] else "FAIL"
        fp_status = "FP" if r['clean_detected'] else "OK"
        
        print(f"{idx:<3} {prompt_short:<45} {wm_status:<11} {fp_status:<6} {r['wm_z_score']:>6.2f}   {r['ppl_increase']:>6.1f}%")
    
    # Final verdict
    print_header("FINAL VERDICT")
    
    success_rate = wm_detected / total_tests
    
    if success_rate >= 0.8 and false_positives == 0:
        verdict = "EXCELLENT - System performing very well"
        status = "READY FOR DEPLOYMENT"
    elif success_rate >= 0.6 and false_positives == 0:
        verdict = "GOOD - System working as expected"
        status = "OPERATIONAL"
    elif success_rate >= 0.5:
        verdict = "MODERATE - Detection could be stronger"
        status = "NEEDS TUNING"
    else:
        verdict = "POOR - Detection too weak"
        status = "REQUIRES REDESIGN"
    
    print(f"\n{verdict}")
    print(f"Status: {status}")
    print(f"\nKey Metrics:")
    print(f"  Detection rate: {100*success_rate:.1f}%")
    print(f"  False positives: {false_positives}")
    print(f"  Average separation: {avg_separation:.2f} standard deviations")
    print(f"  Quality cost: {avg_ppl_increase:.1f}% perplexity increase")
    print(f"\nVisualizations saved to: outputs/")
    
    print("\n" + "=" * 80)
    print("  Multi-prompt testing complete\n")

if __name__ == "__main__":
    main()
