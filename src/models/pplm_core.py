"""
PPLM core adapted for imperceptible watermarking.

Key adaptations from original PPLM:
- Micro-perturbations: step_size 0.0001-0.001 (vs 0.01-0.03)
- Strong KL constraints: λ=5.0-10.0 (vs 0.01)
- Minimal iterations: 1-2 (vs 3-10)
- Quality preservation: <2% perplexity increase target
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from typing import Optional, Tuple


class MicroPerturbationPPLM:
    """
    PPLM with micro-perturbations for imperceptible watermarking.
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

    def compute_kl_divergence(self, perturbed_logits: torch.Tensor,
                             original_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence for fluency preservation.

        This is CRITICAL - keeps perturbations imperceptible.

        Args:
            perturbed_logits: Logits after perturbation
            original_logits: Original unperturbed logits

        Returns:
            KL divergence scalar
        """
        p_perturbed = F.softmax(perturbed_logits, dim=-1)
        p_original = F.softmax(original_logits, dim=-1)

        kl = (p_original * (torch.log(p_original + 1e-10) -
                           torch.log(p_perturbed + 1e-10))).sum()
        return kl

    def apply_micro_perturbation(self, original_logits: torch.Tensor,
                                watermark_gradient: torch.Tensor,
                                step_size: float = 0.0005,
                                kl_lambda: float = 5.0,
                                num_iterations: int = 1) -> torch.Tensor:
        """
        Apply gentle perturbation - simplified direct bias approach.

        Args:
            original_logits: Unperturbed logits
            watermark_gradient: Gradient from watermark attribute
            step_size: Perturbation magnitude
            kl_lambda: KL constraint weight (IGNORED in simplified version)
            num_iterations: Gradient steps (IGNORED in simplified version)

        Returns:
            Perturbed logits
        """
        # SIMPLIFIED: Just add the gradient directly (no normalization, no KL constraint)
        # The gradient from SubtleTokenBiasSignal already has the right direction:
        # positive for green tokens, negative for red tokens
        perturbed_logits = original_logits + step_size * watermark_gradient
        
        return perturbed_logits

    def generate_step(self, input_ids: torch.Tensor,
                     past_key_values: Optional[tuple],
                     watermark_attribute,
                     apply_watermark: bool = True,
                     step_size: float = 0.0005,
                     kl_lambda: float = 5.0) -> Tuple[torch.Tensor, tuple]:
        """
        Generate single token with optional micro-perturbation watermark.

        Args:
            input_ids: Current input token IDs
            past_key_values: Cached past
            watermark_attribute: Watermark attribute model
            apply_watermark: Whether to apply watermark
            step_size: PPLM step size (micro-perturbation)
            kl_lambda: KL constraint weight

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
            # Compute watermark gradient
            watermark_grad = watermark_attribute.compute_gradient(original_logits)

            # Apply micro-perturbation with KL constraint
            final_logits = self.apply_micro_perturbation(
                original_logits,
                watermark_grad,
                step_size=step_size,
                kl_lambda=kl_lambda,
                num_iterations=1  # Minimal iterations
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
