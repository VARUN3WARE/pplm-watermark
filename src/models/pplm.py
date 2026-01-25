"""
Watermark generator using micro-perturbation PPLM.

Adapted for imperceptible watermarking with:
- Micro-perturbations (step_size 0.0005 vs 0.01)
- Strong KL constraints (λ=5.0 vs 0.01)
- Quality preservation target (<2% perplexity increase)
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, List
from .pplm_core import MicroPerturbationPPLM
from ..utils.key_generation import WatermarkKey
from ..watermark.signals import SubtleTokenBiasSignal


class WatermarkGenerator:
    """
    Main interface for imperceptible watermarked text generation.
    """

    def __init__(self, model_name: str = "gpt2", secret_key: str = "default-secret"):
        """
        Initialize watermark generator.

        Args:
            model_name: HuggingFace model identifier
            secret_key: Secret key for watermark generation
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
        
        # Initialize micro-perturbation PPLM
        self.pplm = MicroPerturbationPPLM(self.model, self.device)
        
        # Initialize key manager
        self.key_manager = WatermarkKey(
            secret_key=secret_key,
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=768
        )
        
        # Create subtle watermark attribute (52% target)
        green_tokens = self.key_manager.get_green_tokens()
        
        # CRITICAL: Ensure all green tokens are within vocab bounds
        actual_vocab_size = len(self.tokenizer)
        green_tokens_filtered = {t for t in green_tokens if 0 <= t < actual_vocab_size}
        
        if len(green_tokens_filtered) != len(green_tokens):
            print(f"WARNING: Filtered {len(green_tokens) - len(green_tokens_filtered)} out-of-bounds tokens")
        
        self.watermark_attr = SubtleTokenBiasSignal(
            green_tokens=green_tokens_filtered,
            target_ratio=0.52,  # Gentle 2% bias for imperceptibility
            vocab_size=actual_vocab_size
        )
        
        print(f"Initialized WatermarkGenerator with {model_name} on {self.device}")
        print(f"Vocab size: {actual_vocab_size}, Tokenizer vocab_size: {self.tokenizer.vocab_size}")
        print(f"Green tokens: {len(green_tokens_filtered)} (~{100*len(green_tokens_filtered)/actual_vocab_size:.0f}%)")
        print(f"Target ratio: 52% (subtle 2% bias)")

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        step_size: float = 0.0005,  # Micro-perturbation!
        kl_lambda: float = 5.0,     # Strong KL constraint!
        burst_interval: int = 50,
        burst_length: int = 6,
        temperature: float = 1.0
    ) -> str:
        """
        Generate watermarked text with imperceptible micro-perturbations.

        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            step_size: PPLM step size (0.0005 recommended for imperceptibility)
            kl_lambda: KL constraint weight (5.0 recommended for quality)
            burst_interval: Tokens between burst starts
            burst_length: Number of consecutive watermarked tokens
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Get burst positions
        burst_positions = self.key_manager.get_burst_positions(
            max_length=max_length,
            burst_interval=burst_interval,
            burst_length=burst_length
        )
        
        generated_tokens = []
        past_key_values = None
        
        for position in range(max_length):
            # Check if this position should be watermarked
            apply_watermark = position in burst_positions
            
            # Generate next token with micro-perturbation
            next_token, past_key_values = self.pplm.generate_step(
                input_ids=input_ids if position == 0 else next_token.unsqueeze(0),
                past_key_values=past_key_values,
                watermark_attribute=self.watermark_attr if apply_watermark else None,
                apply_watermark=apply_watermark,
                step_size=step_size,
                kl_lambda=kl_lambda
            )
            
            generated_tokens.append(next_token.item())
            
            # Stop at end of sequence
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode full sequence
        full_sequence = input_ids[0].tolist() + generated_tokens
        return self.tokenizer.decode(full_sequence, skip_special_tokens=True)
    
    def generate_clean(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate without watermark for comparison.

        Args:
            prompt: Input prompt
            max_length: Maximum tokens

        Returns:
            Clean generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                do_sample=True,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"WatermarkGenerator(device={self.device}, target_ratio=0.52)"
