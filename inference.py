#!/usr/bin/env python3
"""
PPLM-Based Text Watermarking: Live Inference Demo

This script demonstrates the complete watermarking pipeline:
- Generates watermarked text using direct logit bias
- Generates clean baseline text for comparison
- Performs statistical detection using z-score analysis
- Evaluates quality impact via perplexity metrics

Usage:
    python inference.py

Author: Research Project
Date: January 2026
"""

import torch
from src.models.pplm import WatermarkGenerator
from src.utils.key_generation import WatermarkKey
from src.watermark.detector import WatermarkDetector
from src.evaluation.quality import QualityEvaluator

def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)

def main():
    print_separator("PPLM WATERMARK SYSTEM - LIVE DEMO")
    
    # Configuration
    SECRET_KEY = "demo-secret-key-2026"
    PROMPT = "The future of artificial intelligence is"
    MAX_LENGTH = 100
    
    print(f"\nConfiguration:")
    print(f"   Secret Key: {SECRET_KEY}")
    print(f"   Prompt: '{PROMPT}'")
    print(f"   Max Length: {MAX_LENGTH} tokens")
    print(f"   Model: GPT-2 (124M)")
    
    # Initialize generator
    print("\nInitializing watermark generator...")
    generator = WatermarkGenerator(
        model_name="gpt2",
        secret_key=SECRET_KEY
    )
    print(f"   Device: {generator.device}")
    
    # Initialize detector
    key_manager = WatermarkKey(SECRET_KEY, generator.tokenizer.vocab_size, 768)
    detector = WatermarkDetector(key_manager, generator.tokenizer)
    
    # Initialize quality evaluator
    evaluator = QualityEvaluator(model_name="gpt2")
    
    # Generate watermarked text
    print_separator("GENERATING WATERMARKED TEXT")
    print("\nGenerating...")
    print(f"   Parameters: step_size=0.5, burst_interval=10, burst_length=15, kl_lambda=0.0")
    
    watermarked_text = generator.generate(
        prompt=PROMPT,
        max_length=MAX_LENGTH,
        step_size=0.5,
        kl_lambda=0.0,
        burst_interval=10,
        burst_length=15
    )
    
    print(f"\nWatermarked Text:")
    print(f"   {watermarked_text}")
    
    # Generate clean text
    print_separator("GENERATING CLEAN TEXT")
    print("\nGenerating...")
    
    clean_text = generator.generate_clean(
        prompt=PROMPT,
        max_length=MAX_LENGTH
    )
    
    print(f"\nClean Text:")
    print(f"   {clean_text}")
    
    # Detect watermark in watermarked text
    print_separator("DETECTION TEST: Watermarked Text")
    
    detected_wm, score_wm, metadata_wm = detector.detect(
        watermarked_text,
        generator.model,
        token_weight=1.0,
        semantic_weight=0.0,
        threshold=2.0
    )
    
    green_ratio_wm = metadata_wm['token']['green_ratio']
    z_score_wm = metadata_wm['token']['z_score']
    
    print(f"\nResults:")
    print(f"   Green Token Ratio: {green_ratio_wm*100:.1f}%")
    print(f"   Z-score: {z_score_wm:.2f}")
    print(f"   Combined Score: {score_wm:.2f}")
    print(f"   Threshold: 2.0")
    print(f"   Detection: {'WATERMARK DETECTED' if detected_wm else 'NOT DETECTED'}")
    
    # Detect watermark in clean text
    print_separator("DETECTION TEST: Clean Text")
    
    detected_clean, score_clean, metadata_clean = detector.detect(
        clean_text,
        generator.model,
        token_weight=1.0,
        semantic_weight=0.0,
        threshold=2.0
    )
    
    green_ratio_clean = metadata_clean['token']['green_ratio']
    z_score_clean = metadata_clean['token']['z_score']
    
    print(f"\nResults:")
    print(f"   Green Token Ratio: {green_ratio_clean*100:.1f}%")
    print(f"   Z-score: {z_score_clean:.2f}")
    print(f"   Detection: {'FALSE POSITIVE' if detected_clean else 'NOT DETECTED (Correct)'}")
    
    # Quality evaluation
    print_separator("QUALITY EVALUATION")
    
    print("\nComputing perplexity...")
    quality_results = evaluator.compute_perplexity_increase(watermarked_text, clean_text)
    ppl_wm = quality_results['watermarked_perplexity']
    ppl_clean = quality_results['clean_perplexity']
    increase = quality_results['percentage_increase']
    
    print(f"\nPerplexity Analysis:")
    print(f"   Clean Text: {ppl_clean:.2f}")
    print(f"   Watermarked Text: {ppl_wm:.2f}")
    print(f"   Increase: {increase:.1f}%")
    
    # Summary
    print_separator("SUMMARY")
    
    print(f"\nSystem Performance:")
    print(f"   Watermarked Detection: {'PASS' if detected_wm else 'FAIL'}")
    print(f"   Clean Detection: {'PASS' if not detected_clean else 'FAIL (False Positive)'}")
    print(f"   Signal Strength: {(green_ratio_wm - green_ratio_clean)*100:.1f}% difference")
    print(f"   Z-score Separation: {z_score_wm - z_score_clean:.2f} standard deviations")
    
    print(f"\nVerdict:")
    if detected_wm and not detected_clean:
        print(f"   WATERMARK SYSTEM WORKING")
        print(f"   Successfully embedded and detected watermark")
        print(f"   No false positives on clean text")
    elif detected_wm and detected_clean:
        print(f"   Watermark detected but FALSE POSITIVE on clean text")
    elif not detected_wm and not detected_clean:
        print(f"   Watermark NOT detected (signal too weak)")
    else:
        print(f"   System malfunction")
    
    print_separator()
    print("\nDemo complete\n")

if __name__ == "__main__":
    main()
