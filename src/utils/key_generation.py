"""
Secret key generation and watermark parameter derivation.

This module provides functionality to derive all watermark parameters
deterministically from a secret key string.
"""

import hashlib
import numpy as np
from typing import Set, List


class WatermarkKey:
    """
    Manages secret key and derives watermark parameters.

    Attributes:
        secret_key: String secret key
        vocab_size: Size of model vocabulary
        embedding_dim: Dimension of model embeddings
    """

    def __init__(self, secret_key: str, vocab_size: int, embedding_dim: int):
        """
        Initialize watermark key manager.

        Args:
            secret_key: Secret key string for deterministic parameter generation
            vocab_size: Size of the model's vocabulary
            embedding_dim: Dimension of the model's embeddings
        """
        self.secret_key = secret_key
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def get_green_tokens(self) -> Set[int]:
        """
        Generate deterministic green token set from secret key.
        Uses SHA256 hash to partition vocabulary 50-50.

        Returns:
            Set of token IDs classified as green (exactly 50%)
        """
        green_tokens = set()
        for token_id in range(self.vocab_size):
            hash_input = f"{self.secret_key}:token:{token_id}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            # Use first 16 hex digits for decision (50/50 split)
            if int(hash_value[:16], 16) % 2 == 0:
                green_tokens.add(token_id)
        return green_tokens

    def get_semantic_direction(self) -> np.ndarray:
        """
        Generate deterministic secret direction vector in embedding space.

        Returns:
            Normalized direction vector of shape (embedding_dim,)
        """
        # Create deterministic seed from secret key
        seed = int(hashlib.sha256(self.secret_key.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed % (2**32))
        
        # Generate random direction
        direction = rng.randn(self.embedding_dim)
        
        # Normalize to unit vector
        direction = direction / np.linalg.norm(direction)
        
        return direction

    def get_burst_positions(
        self,
        max_length: int,
        burst_interval: int = 50,
        burst_length: int = 8
    ) -> List[int]:
        """
        Generate deterministic burst positions with optional jitter.

        Args:
            max_length: Maximum sequence length
            burst_interval: Tokens between burst starts
            burst_length: Number of consecutive watermarked tokens

        Returns:
            List of token positions to watermark
        """
        positions = []
        
        # Create deterministic seed for burst scheduling
        seed = int(hashlib.sha256(f"{self.secret_key}_burst".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed % (2**32))

        current_pos = 0
        while current_pos < max_length:
            # Add small random jitter to burst positions
            jitter = rng.randint(-5, 6)
            start = max(0, current_pos + jitter)
            end = min(max_length, start + burst_length)
            
            # Add all positions in this burst
            positions.extend(range(start, end))
            
            # Move to next burst
            current_pos += burst_interval

        return sorted(set(positions))

    def get_red_tokens(self) -> Set[int]:
        """
        Get the complement set of green tokens (red tokens).

        Returns:
            Set of token IDs classified as red
        """
        green = self.get_green_tokens()
        return set(range(self.vocab_size)) - green

    def __repr__(self) -> str:
        """String representation of the key manager."""
        return (f"WatermarkKey(vocab_size={self.vocab_size}, "
                f"embedding_dim={self.embedding_dim})")
