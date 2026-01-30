#!/usr/bin/env python3
"""
Watermark Detector

Detects watermarks in text files.
This demonstrates the detection phase of the watermarking system.

Usage:
    python detect_watermark.py

Input:
    - watermarked_output.txt: Text to check for watermark
    - clean_output.txt: Clean text for comparison

Author: Research Project
Date: January 2026
"""

import torch
import os
from src.models.pplm import WatermarkGenerator
from src.utils.key_generation import WatermarkKey
from src.watermark.detector import WatermarkDetector
from src.evaluation.quality import QualityEvaluator

def detect_text(filename, detector, model, evaluator=None):
    """Detect watermark in a text file"""
    
    if not os.path.exists(filename):
        print(f"  ERROR: File '{filename}' not found")
        return None
    
    # Read text
    with open(filename, "r") as f:
        text = f.read().strip()
    
    print(f"\nAnalyzing: {filename}")
    print(f"  Text: {text[:100]}...")
    
    # Detect watermark
    detected, score, metadata = detector.detect(
        text,
        model,
        token_weight=1.0,
        semantic_weight=0.0,
        threshold=2.0
    )
    
    green_ratio = metadata['token']['green_ratio']
    z_score = metadata['token']['z_score']
    
    print(f"\nDetection Results:")
    print(f"  Green Token Ratio: {green_ratio*100:.1f}%")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Threshold: 2.0")
    print(f"  Status: {'WATERMARK DETECTED' if detected else 'NO WATERMARK'}")
    
    # Compute perplexity if evaluator provided
    if evaluator:
        ppl = evaluator.compute_perplexity(text)
        print(f"  Perplexity: {ppl:.2f}")
    
    return {
        'detected': detected,
        'z_score': z_score,
        'green_ratio': green_ratio
    }

def main():
    print("=" * 70)
    print("  WATERMARK DETECTOR")
    print("=" * 70)
    
    # Configuration
    SECRET_KEY = "research-watermark-2026"
    
    print(f"\nConfiguration:")
    print(f"  Secret Key: {SECRET_KEY}")
    print(f"  Detection Threshold: 2.0 (z-score)")
    
    # Initialize system
    print("\nInitializing detection system...")
    generator = WatermarkGenerator(model_name="gpt2", secret_key=SECRET_KEY)
    key_manager = WatermarkKey(SECRET_KEY, generator.tokenizer.vocab_size, 768)
    detector = WatermarkDetector(key_manager, generator.tokenizer)
    evaluator = QualityEvaluator(model_name="gpt2")
    
    print(f"  Device: {generator.device}")
    
    # Detect watermark in files
    print("\n" + "=" * 70)
    print("DETECTION TESTS")
    print("=" * 70)
    
    # Test 1: Watermarked text
    print("\n" + "-" * 70)
    print("TEST 1: Watermarked Text")
    print("-" * 70)
    result_wm = detect_text("watermarked_output.txt", detector, generator.model, evaluator)
    
    # Test 2: Clean text
    print("\n" + "-" * 70)
    print("TEST 2: Clean Text")
    print("-" * 70)
    result_clean = detect_text("clean_output.txt", detector, generator.model, evaluator)
    
    # Summary
    if result_wm and result_clean:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        print(f"\nWatermarked Text:")
        print(f"  Detection: {'PASS' if result_wm['detected'] else 'FAIL'}")
        print(f"  Z-score: {result_wm['z_score']:.2f}")
        
        print(f"\nClean Text:")
        print(f"  Detection: {'PASS (correctly not detected)' if not result_clean['detected'] else 'FAIL (false positive)'}")
        print(f"  Z-score: {result_clean['z_score']:.2f}")
        
        separation = result_wm['z_score'] - result_clean['z_score']
        print(f"\nStatistical Separation: {separation:.2f} standard deviations")
        
        print("\nVerdict:")
        if result_wm['detected'] and not result_clean['detected']:
            print("  System working correctly")
            print("  Watermark successfully detected")
            print("  No false positives")
        else:
            print("  System needs adjustment")
    
    print("\n" + "=" * 70)
    print()

if __name__ == "__main__":
    main()
