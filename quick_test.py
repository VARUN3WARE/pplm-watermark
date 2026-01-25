#!/usr/bin/env python3
"""Quick test to verify watermark parameters"""

from src.models.pplm import WatermarkGenerator

generator = WatermarkGenerator(secret_key="test")

# Test generation with explicit strong parameters
text = generator.generate(
    prompt="Hello world",
    max_length=50,
    step_size=0.5,  # STRONG
    burst_interval=10,
    burst_length=15
)

print(f"Generated: {text[:100]}...")
print(f"\nParameters used: step_size=0.5, burst_interval=10, burst_length=15")
