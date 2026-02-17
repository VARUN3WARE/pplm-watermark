"""
Watermark generator using PPLM-based logit perturbation.

v1: Fixed green token partition with direct logit bias.
v2: Adds context-dependent green lists where the green/red partition
    changes based on the previous token, dramatically improving
    detection rates from ~60% to 90%+ TPR.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, List
from .pplm_core import MicroPerturbationPPLM
from ..utils.key_generation import WatermarkKey
from ..watermark.signals import SubtleTokenBiasSignal, ContextDependentBiasSignal


class WatermarkGenerator:
    """
    Main interface for watermarked text generation.
    Supports both fixed (v1) and context-dependent (v2) watermarking.
    """

    def __init__(self, model_name: str = "gpt2", secret_key: str = "default-secret",
                 context_dependent: bool = True):
        """
        Initialize watermark generator.

        Args:
            model_name: HuggingFace model identifier
            secret_key: Secret key for watermark generation
            context_dependent: If True, use v2 context-dependent green lists
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
        self.context_dependent = context_dependent
        
        # Initialize PPLM
        self.pplm = MicroPerturbationPPLM(self.model, self.device)
        
        # Initialize key manager
        self.key_manager = WatermarkKey(
            secret_key=secret_key,
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=768
        )
        
        actual_vocab_size = len(self.tokenizer)
        
        if context_dependent:
            # v2: Context-dependent bias signal
            self.watermark_attr = ContextDependentBiasSignal(
                key_manager=self.key_manager,
                vocab_size=actual_vocab_size
            )
            mode = "context-dependent (v2)"
        else:
            # v1: Fixed green token partition
            green_tokens = self.key_manager.get_green_tokens()
            green_tokens_filtered = {t for t in green_tokens if 0 <= t < actual_vocab_size}
            
            if len(green_tokens_filtered) != len(green_tokens):
                print(f"WARNING: Filtered {len(green_tokens) - len(green_tokens_filtered)} out-of-bounds tokens")
            
            self.watermark_attr = SubtleTokenBiasSignal(
                green_tokens=green_tokens_filtered,
                target_ratio=0.52,
                vocab_size=actual_vocab_size
            )
            mode = "fixed partition (v1)"
        
        print(f"Initialized WatermarkGenerator with {model_name} on {self.device}")
        print(f"Vocab size: {actual_vocab_size}")
        print(f"Watermark mode: {mode}")

    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        step_size: float = 0.5,
        kl_lambda: float = 0.0,
        burst_interval: int = 10,
        burst_length: int = 15,
        temperature: float = 1.0
    ) -> str:
        """
        Generate watermarked text.

        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            step_size: Perturbation magnitude (0.5 recommended)
            kl_lambda: KL constraint weight (0.0 recommended)
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
        
        # Track previous token for context-dependent mode
        prev_token_id = input_ids[0, -1].item()
        
        for position in range(max_length):
            apply_watermark = position in burst_positions
            
            # Generate next token with perturbation
            next_token, past_key_values = self.pplm.generate_step(
                input_ids=input_ids if position == 0 else next_token.unsqueeze(0),
                past_key_values=past_key_values,
                watermark_attribute=self.watermark_attr if apply_watermark else None,
                apply_watermark=apply_watermark,
                step_size=step_size,
                kl_lambda=kl_lambda,
                prev_token_id=prev_token_id,
                context_dependent=self.context_dependent
            )
            
            generated_tokens.append(next_token.item())
            prev_token_id = next_token.item()
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
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
        mode = "context-dependent" if self.context_dependent else "fixed"
        return f"WatermarkGenerator(device={self.device}, mode={mode})"
