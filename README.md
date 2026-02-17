# PPLM-Based Text Watermarking System

A research implementation of statistical text watermarking using Plug and Play Language Models (PPLM). This system embeds detectable watermarks in AI-generated text through direct logit bias during inference, without modifying the base language model.

## Academic Context

This project implements a novel approach to text watermarking based on PPLM (Dathathri et al., 2019), adapted specifically for imperceptible watermark embedding. The system demonstrates how controlled perturbations in the generation process can create statistically detectable patterns while maintaining text coherence.

**Key Innovation (v2)**: Context-dependent green lists where the green/red token partition changes based on the previous token (following Kirchenbauer et al., 2023), combined with PPLM-based logit bias for watermark injection. This achieves 80% TPR with 0% FPR.

## Key Features

- Statistical Watermarking: Embeds detectable patterns without modifying the base model
- Context-Dependent Green Lists (v2): Token partition changes per position for stronger detection
- Direct Logit Bias: Simple and effective perturbation approach via PPLM
- Burst Scheduling: Intermittent watermarking for quality preservation
- Secret Key Based: Deterministic watermark derivation from secret key
- Visualization Suite: ROC curves, score distributions, green token heatmaps
- Robustness Testing: Evaluates watermark survival under text modifications

## Performance

### v2 Results (Context-Dependent, 10 prompts, 80 tokens each)

| Metric                          | Value               |
| ------------------------------- | ------------------- |
| True Positive Rate              | 80%                 |
| False Positive Rate             | 0%                  |
| Avg Green Ratio (Watermarked)   | 67.0%               |
| Avg Green Ratio (Clean)         | 49.9%               |
| Detection Separation            | 3.12 std deviations |
| Signal Strength                 | 17.1% difference    |

### Robustness (v2, 8 prompts, 100 tokens each)

| Attack             | Low    | Medium | High   | Extreme |
| ------------------ | ------ | ------ | ------ | ------- |
| Truncation         | 100%   | 100%   | 75%    | 12%     |
| Word Deletion      | 100%   | 100%   | 100%   | 50%     |
| Word Insertion     | 100%   | 100%   | 100%   | 100%    |
| Word Swap          | 100%   | 100%   | 100%   | 88%     |

## Quick Start

### Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Basic Usage (v2)

```python
from src.models.pplm import WatermarkGenerator
from src.utils.key_generation import WatermarkKey
from src.watermark.detector import WatermarkDetector

# Initialize generator with context-dependent mode (v2)
SECRET_KEY = "your-secret-key"
generator = WatermarkGenerator(
    model_name="gpt2",
    secret_key=SECRET_KEY,
    context_dependent=True  # v2 mode
)

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
detector = WatermarkDetector(
    key_manager, generator.tokenizer,
    context_dependent=True  # Must match generator mode
)
detected, score, metadata = detector.detect(
    watermarked_text,
    threshold=2.0
)
```

## Running Tests

```bash
# Two-step workflow (recommended for demonstration)
python watermark_text.py    # Step 1: Generate watermarked text
python detect_watermark.py  # Step 2: Detect watermark in generated text

# All-in-one demo with visualizations
python inference.py

# Multi-prompt evaluation (10 prompts, generates plots)
python test_multiple_prompts.py

# Robustness testing (5 attack types, generates plots)
python robustness_test.py
```

### Two-Step Workflow

The watermarking system operates in two independent phases:

**Step 1: Generate Watermarked Text**
```bash
python watermark_text.py
```
- Generates watermarked text using the secret key
- Saves output to `watermarked_output.txt`
- Also generates clean text for comparison in `clean_output.txt`

**Step 2: Detect Watermark**
```bash
python detect_watermark.py
```
- Reads text files and checks for watermark presence
- Uses the same secret key to verify watermark
- Shows detection statistics (z-score, green token ratio)
- Compares watermarked vs clean text

## Working Parameters

- Step Size: 0.5 (logit bias strength)
- Burst Interval: 10 tokens
- Burst Length: 15 tokens
- Detection Threshold: 2.0 (z-score)
- Context Dependent: True (v2, recommended)

## System Architecture

1. Key Generation: SHA256-based token partitioning
2. Green List (v2): Context-dependent partition via hash(secret_key, prev_token)
3. Watermark Embedding: Direct logit bias (+0.5 for green tokens) via PPLM
4. Detection: Z-score statistical test on context-dependent green token ratio
5. Result: 80% detection rate with 0% false positives

### v1 vs v2 Comparison

| Feature | v1 (Fixed) | v2 (Context-Dependent) |
| ------- | ---------- | ---------------------- |
| Green list | Same for all positions | Changes per token |
| TPR | 60% | 80% |
| FPR | 0% | 0% |
| Robustness | Not tested | 5 attack types tested |
| Visualization | None | ROC + distributions + heatmaps |

## Experimental History

- Aggressive parameters (0.05-0.20): Failed
- Micro-perturbations (0.0005) + KL constraints: Too weak
- 70/30 token ratio: Aligned with natural distribution (failed)
- Direct bias (0.5) + 50/50 split (v1): 60% TPR
- Context-dependent green lists (v2): 80% TPR

## Project Files

- **watermark_text.py** - Generate watermarked text (Step 1)
- **detect_watermark.py** - Detect watermarks in text (Step 2)
- **inference.py** - All-in-one demonstration with visualizations
- **test_multiple_prompts.py** - Multi-prompt evaluation with plots
- **robustness_test.py** - Robustness testing against text attacks
- **TECHNICAL_REPORT.md** - Complete research documentation
- **src/evaluation/visualize.py** - Visualization module (ROC, heatmaps)

## References

- Dathathri, S., et al. (2019). "Plug and Play Language Models: A Simple Approach to Controlled Text Generation." ICLR 2020.
- Kirchenbauer, J., et al. (2023). "A Watermark for Large Language Models." ICML 2023.
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Status**: v2 Complete | **Updated**: February 2026
