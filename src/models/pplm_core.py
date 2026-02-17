"""
PPLM core adapted for imperceptible watermarking.

v1: Direct logit bias with fixed green token partition.
v2: Supports context-dependent bias where green set changes per token.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from typing import Optional, Tuple


class MicroPerturbationPPLM:
    """
    PPLM with micro-perturbations for imperceptible watermarking.
    Supports both fixed (v1) and context-dependent (v2) bias signals.
    """

    def __init__(self, model: GPT2LMHeadModel, device: torch.device):
        """
        Initialize micro-perturbation PPLM.

        Args:
            model: Pre-trained language model
            device: torch device
        """
        self.model = model
        self.device = device
        self.model.eval()

    def apply_micro_perturbation(self, original_logits: torch.Tensor,
                                watermark_gradient: torch.Tensor,
                                step_size: float = 0.5) -> torch.Tensor:
        """
        Apply direct bias perturbation to logits.

        Args:
            original_logits: Unperturbed logits
            watermark_gradient: Bias vector from watermark signal
            step_size: Perturbation magnitude

        Returns:
            Perturbed logits
        """
        perturbed_logits = original_logits + step_size * watermark_gradient
        return perturbed_logits

    def generate_step(self, input_ids: torch.Tensor,
                     past_key_values: Optional[tuple],
                     watermark_attribute,
                     apply_watermark: bool = True,
                     step_size: float = 0.5,
                     kl_lambda: float = 0.0,
                     prev_token_id: Optional[int] = None,
                     context_dependent: bool = False) -> Tuple[torch.Tensor, tuple]:
        """
        Generate single token with optional watermark perturbation.

        Args:
            input_ids: Current input token IDs
            past_key_values: Cached past
            watermark_attribute: Watermark signal (SubtleTokenBiasSignal or ContextDependentBiasSignal)
            apply_watermark: Whether to apply watermark
            step_size: Perturbation magnitude
            kl_lambda: KL constraint weight (unused in simplified version)
            prev_token_id: Previous token ID (for context-dependent mode)
            context_dependent: Whether to use context-dependent bias

        Returns:
            Tuple of (next_token, updated_past_key_values)
        """
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

        original_logits = outputs.logits[:, -1, :].squeeze(0)
        updated_past = outputs.past_key_values

        if apply_watermark and watermark_attribute is not None:
            # Compute watermark bias vector
            if context_dependent and prev_token_id is not None:
                watermark_grad = watermark_attribute.compute_gradient(
                    original_logits, prev_token_id=prev_token_id
                )
            else:
                watermark_grad = watermark_attribute.compute_gradient(original_logits)

            # Apply perturbation
            final_logits = self.apply_micro_perturbation(
                original_logits,
                watermark_grad,
                step_size=step_size
            )
        else:
            final_logits = original_logits

        # Ensure 2D for multinomial
        if final_logits.dim() == 1:
            final_logits = final_logits.unsqueeze(0)

        # Sample next token
        probs = F.softmax(final_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token, updated_past
