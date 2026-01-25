# PPLM-Based Text Watermarking System

A working implementation of statistical text watermarking using Plug and Play Language Models (PPLM). This system embeds imperceptible watermarks in AI-generated text through direct logit bias during inference.

## 🎯 Key Features

- **Statistical Watermarking**: Embeds detectable patterns without modifying the base model
- **Direct Logit Bias**: Simple and effective perturbation approach
- **Burst Scheduling**: Intermittent watermarking for robustness
- **Quality Preservation**: Minimal perplexity increase
- **Secret Key Based**: Deterministic watermark derivation from secret key

## 📊 Performance

Based on multi-sample testing (10 samples, 100 tokens each):

| Metric                              | Value               |
| ----------------------------------- | ------------------- |
| **True Positive Rate**              | 70%                 |
| **False Positive Rate**             | 0%                  |
| **Green Token Ratio (Watermarked)** | 67.6% ± 2.8%        |
| **Green Token Ratio (Clean)**       | 53.7% ± 4.8%        |
| **Detection Separation**            | 2.20 std deviations |
| **Signal Strength**                 | 13.9% difference    |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment with uv
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
    step_size=0.5,       # Logit bias strength
    burst_interval=10,   # Watermark every 10 tokens
    burst_length=15      # For 15 consecutive tokens
)

# Detect watermark
key_manager = WatermarkKey(SECRET_KEY, generator.tokenizer.vocab_size, 768)
detector = WatermarkDetector(key_manager, generator.tokenizer)
detected, score, metadata = detector.detect(watermarked_text, generator.model, threshold=2.0)
```

## 🧪 Running Tests

```bash
# Single sample test
python test_simple.py

# Multi-sample evaluation
python test_multi_sample.py

# Full example with quality assessment
python examples/basic_usage.py
```

## 📈 Working Parameters

- **Step Size**: 0.5 (logit bias strength)
- **Burst Interval**: 10 tokens
- **Burst Length**: 15 tokens
- **Threshold**: 2.0 (z-score)

## 🔬 How It Works

1. **Key Generation**: SHA256-based 50/50 token partitioning
2. **Embedding**: Direct logit bias (+0.5 for green tokens)
3. **Detection**: Z-score test on green token ratio
4. **Success**: 70% detection rate with 0% false positives

## 📝 Key Lessons Learned

- ❌ Aggressive parameters (0.05-0.20) → Failed
- ❌ Micro-perturbations (0.0005) + KL constraints → Too weak
- ❌ 70/30 token ratio → Aligned with natural distribution
- ✅ **Direct bias (0.5) + 50/50 split → Success!**
