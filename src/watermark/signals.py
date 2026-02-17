"""
Watermark signal implementations.

This module implements different watermark signals that can be used
to steer text generation via PPLM.

v2: Adds ContextDependentBiasSignal where the bias vector changes
based on the previous token, providing much stronger detection signal.
"""

import torch
import torch.nn as nn
from typing import Set, Optional


class SubtleTokenBiasSignal:
    """
    Creates gentle bias toward green tokens for imperceptible watermark.
    
    SIMPLIFIED: Direct logit bias instead of gradient-based.
    Uses FIXED green token partition (v1 compatibility).
    """

    def __init__(self, green_tokens: Set[int], target_ratio: float = 0.52, vocab_size: int = 50257):
        """
        Initialize subtle token bias signal.

        Args:
            green_tokens: Set of token IDs to bias toward
            target_ratio: Target probability mass on green tokens (UNUSED in direct bias)
            vocab_size: Vocabulary size
        """
        self.green_tokens = green_tokens
        self.target_ratio = target_ratio
        self.vocab_size = vocab_size
        
        # Create direct bias vector
        self.bias_vector = torch.zeros(vocab_size)
        for token_id in green_tokens:
            if 0 <= token_id < vocab_size:
                self.bias_vector[token_id] = 1.0
        # Normalize so mean is 0 (preserve total probability mass)
        green_fraction = len([t for t in green_tokens if 0 <= t < vocab_size]) / vocab_size
        for token_id in range(vocab_size):
            if token_id not in green_tokens:
                self.bias_vector[token_id] = -green_fraction / (1 - green_fraction)
        
        self.green_mask = None

    def _create_mask(self, vocab_size: int, device: torch.device):
        """Create binary mask for green tokens."""
        if self.green_mask is None or len(self.green_mask) != vocab_size:
            mask = torch.zeros(vocab_size, device=device)
            for token_id in self.green_tokens:
                if 0 <= token_id < vocab_size:
                    mask[token_id] = 1.0
            self.green_mask = mask
        return self.green_mask.to(device)

    def compute_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute expected green ratio from current logits.

        Args:
            logits: Token logits of shape (vocab_size,)

        Returns:
            Scalar score (expected green mass)
        """
        probs = torch.softmax(logits, dim=-1)
        mask = self._create_mask(len(logits), logits.device)
        green_mass = (probs * mask).sum()
        return green_mass

    def compute_gradient(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return direct bias vector (not a gradient, but used as perturbation).

        Args:
            logits: Token logits (shape info only)

        Returns:
            Bias vector of same shape as logits
        """
        return self.bias_vector.to(logits.device)


class ContextDependentBiasSignal:
    """
    Context-dependent watermark signal (v2).
    
    The bias vector changes based on the previous token, following
    Kirchenbauer et al. (2023). This makes the watermark signal much
    stronger for detection because consecutive green tokens in natural
    text become exponentially unlikely.
    """

    def __init__(self, key_manager, vocab_size: int = 50257):
        """
        Initialize context-dependent bias signal.

        Args:
            key_manager: WatermarkKey instance for green token computation
            vocab_size: Vocabulary size
        """
        self.key_manager = key_manager
        self.vocab_size = vocab_size
        # Cache bias vectors to avoid recomputation
        self._bias_cache = {}

    def _get_bias_vector(self, prev_token_id: int) -> torch.Tensor:
        """
        Compute bias vector conditioned on previous token.

        Args:
            prev_token_id: Previous token ID

        Returns:
            Bias vector of shape (vocab_size,)
        """
        if prev_token_id in self._bias_cache:
            return self._bias_cache[prev_token_id]

        green_tokens = self.key_manager.get_green_tokens_context(prev_token_id)
        
        bias = torch.zeros(self.vocab_size)
        green_fraction = len(green_tokens) / self.vocab_size
        
        for token_id in range(self.vocab_size):
            if token_id in green_tokens:
                bias[token_id] = 1.0
            else:
                bias[token_id] = -green_fraction / (1 - green_fraction)

        self._bias_cache[prev_token_id] = bias
        return bias

    def compute_gradient(self, logits: torch.Tensor, prev_token_id: int = 0) -> torch.Tensor:
        """
        Return context-dependent bias vector.

        Args:
            logits: Token logits (for device info)
            prev_token_id: Previous token ID for context

        Returns:
            Bias vector of same shape as logits
        """
        bias = self._get_bias_vector(prev_token_id)
        return bias.to(logits.device)

    def clear_cache(self):
        """Clear cached bias vectors."""
        self._bias_cache = {}


class SemanticDirectionSignal:
    """
    Implements semantic direction watermark signal.
    Nudges hidden states toward secret direction vector.
    """

    def __init__(self, direction_vector: torch.Tensor, strength: float = 1.0):
        """
        Initialize semantic direction signal.

        Args:
            direction_vector: Target direction in embedding space
            strength: Multiplier for gradient strength
        """
        self.direction = direction_vector.float()
        self.strength = strength

    def compute_score(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity to secret direction.

        Args:
            hidden_state: Hidden state vector of shape (hidden_dim,) or (1, hidden_dim)

        Returns:
            Scalar similarity score
        """
        # Squeeze to ensure 1D tensor
        if hidden_state.dim() > 1:
            hidden_state = hidden_state.squeeze()
        
        # Move direction to same device as hidden_state
        direction = self.direction.to(hidden_state.device)
        
        # Normalize both vectors
        hidden_norm = hidden_state / (hidden_state.norm() + 1e-8)
        direction_norm = direction / (direction.norm() + 1e-8)
        
        # Compute cosine similarity
        score = torch.dot(hidden_norm, direction_norm)
        
        return score * self.strength

    def compute_gradient(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of similarity with respect to hidden state.

        Args:
            hidden_state: Hidden state requiring gradient

        Returns:
            Gradient tensor of same shape as hidden_state
        """
        hidden_state = hidden_state.clone().detach().requires_grad_(True)
        score = self.compute_score(hidden_state)
        score.backward()
        return hidden_state.grad.clone()

