#!/usr/bin/env python3
"""
Watermark Text Generator

Generates watermarked text and saves it to a file.
This demonstrates the text generation phase of the watermarking system.

Usage:
    python watermark_text.py

Output:
    - watermarked_output.txt: Generated watermarked text
    - clean_output.txt: Clean text for comparison

Author: Research Project
Date: January 2026
"""

import torch
from src.models.pplm import WatermarkGenerator

def main():
    print("=" * 70)
    print("  WATERMARK TEXT GENERATOR")
    print("=" * 70)
    
    # Configuration
    SECRET_KEY = "research-watermark-2026"
    PROMPT = "The future of artificial intelligence is"
    MAX_LENGTH = 100
    
    print(f"\nConfiguration:")
    print(f"  Secret Key: {SECRET_KEY}")
    print(f"  Prompt: '{PROMPT}'")
    print(f"  Max Length: {MAX_LENGTH} tokens")
    print(f"  Model: GPT-2")
    
    # Initialize generator
    print("\nInitializing watermark generator...")
    generator = WatermarkGenerator(
        model_name="gpt2",
        secret_key=SECRET_KEY
    )
    print(f"  Device: {generator.device}")
    
    # Generate watermarked text
    print("\n" + "-" * 70)
    print("Generating watermarked text...")
    print(f"Parameters: step_size=0.5, burst_interval=10, burst_length=15")
    
    watermarked_text = generator.generate(
        prompt=PROMPT,
        max_length=MAX_LENGTH,
        step_size=0.5,
        kl_lambda=0.0,
        burst_interval=10,
        burst_length=15
    )
    
    # Generate clean text for comparison
    print("\nGenerating clean text (for comparison)...")
    clean_text = generator.generate_clean(
        prompt=PROMPT,
        max_length=MAX_LENGTH
    )
    
    # Save to files
    with open("watermarked_output.txt", "w") as f:
        f.write(watermarked_text)
    
    with open("clean_output.txt", "w") as f:
        f.write(clean_text)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nWatermarked Text:")
    print(f"  {watermarked_text}")
    print(f"\nSaved to: watermarked_output.txt")
    
    print("\nClean Text (for comparison):")
    print(f"  {clean_text}")
    print(f"\nSaved to: clean_output.txt")
    
    print("\n" + "=" * 70)
    print("Next step: Run 'python detect_watermark.py' to detect watermarks")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
