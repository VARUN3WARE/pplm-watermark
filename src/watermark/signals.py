"""
Watermark signal implementations.

This module implements different watermark signals that can be used
to steer text generation via PPLM.
"""

import torch
import torch.nn as nn
from typing import Set


class SubtleTokenBiasSignal:
    """
    Creates gentle bias toward green tokens for imperceptible watermark.
    
    SIMPLIFIED: Direct logit bias instead of gradient-based.
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
                # Critical: bounds checking!
                if 0 <= token_id < vocab_size:
                    mask[token_id] = 1.0
            self.green_mask = mask
        return self.green_mask.to(device)  # Ensure on correct device

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

