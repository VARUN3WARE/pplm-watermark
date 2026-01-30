# PPLM-Based Text Watermarking System

A research implementation of statistical text watermarking using Plug and Play Language Models (PPLM). This system embeds detectable watermarks in AI-generated text through direct logit bias during inference, without modifying the base language model.

## Academic Context

This project implements a novel approach to text watermarking based on PPLM (Dathathri et al., 2019), adapted specifically for imperceptible watermark embedding. The system demonstrates how controlled perturbations in the generation process can create statistically detectable patterns while maintaining text coherence.

**Key Innovation**: Direct logit bias with 50/50 token partitioning achieves superior detection rates compared to gradient-based micro-perturbations.

## Key Features

- Statistical Watermarking: Embeds detectable patterns without modifying the base model
- Direct Logit Bias: Simple and effective perturbation approach
- Burst Scheduling: Intermittent watermarking for robustness
- Secret Key Based: Deterministic watermark derivation from secret key

## Performance

Based on multi-sample testing (10 samples, 100 tokens each):

| Metric                          | Value               |
| ------------------------------- | ------------------- |
| True Positive Rate              | 70%                 |
| False Positive Rate             | 0%                  |
| Green Token Ratio (Watermarked) | 67.6% ± 2.8%        |
| Green Token Ratio (Clean)       | 53.7% ± 4.8%        |
| Detection Separation            | 2.20 std deviations |
| Signal Strength                 | 13.9% difference    |

## Quick Start

### Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Basic Usage

```python
from src.models.pplm import WatermarkGenerator
from src.utils.key_generation import WatermarkKey
from src.watermark.detector import WatermarkDetector

# Initialize generator
SECRET_KEY = "your-secret-key"
generator = WatermarkGenerator(model_name="gpt2", secret_key=SECRET_KEY)

# Generate watermarked text
watermarked_text = generator.generate(
    prompt="The future of AI is",
    max_length=100,
    step_size=0.5,
    kl_lambda=0.0,
    burst_interval=10,
    burst_length=15
)

# Detect watermark
key_manager = WatermarkKey(SECRET_KEY, generator.tokenizer.vocab_size, 768)
detector = WatermarkDetector(key_manager, generator.tokenizer)
detected, score, metadata = detector.detect(
    watermarked_text,
    generator.model,
    token_weight=1.0,
    semantic_weight=0.0,
    threshold=2.0
)
```

## Running Tests

```bash
# Quick inference demo
python inference.py

# Multi-prompt evaluation
python test_multiple_prompts.py
```

## Working Parameters

- Step Size: 0.5 (logit bias strength)
- Burst Interval: 10 tokens
- Burst Length: 15 tokens
- Detection Threshold: 2.0 (z-score)

## System Architecture

1. Key Generation: SHA256-based 50/50 token partitioning
2. Watermark Embedding: Direct logit bias (+0.5 for green tokens)
3. Detection: Z-score statistical test on green token ratio
4. Result: 70% detection rate with 0% false positives

## Experimental Results

- Aggressive parameters (0.05-0.20): Failed
- Micro-perturbations (0.0005) + KL constraints: Too weak
- 70/30 token ratio: Aligned with natural distribution (failed)
- Direct bias (0.5) + 50/50 split: Success

## References

- Dathathri, S., et al. (2019). "Plug and Play Language Models: A Simple Approach to Controlled Text Generation." ICLR 2020.
- Kirchenbauer, J., et al. (2023). "A Watermark for Large Language Models." ICML 2023.
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI.
