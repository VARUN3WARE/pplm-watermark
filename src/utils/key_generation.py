"""
Secret key generation and watermark parameter derivation.

This module provides functionality to derive all watermark parameters
deterministically from a secret key string.

v2: Adds context-dependent green lists where the green/red partition
changes based on the previous token, following Kirchenbauer et al. (2023).
This significantly improves detection by making accidental green token
runs in natural text much less likely.
"""

import hashlib
import numpy as np
from typing import Set, List
from functools import lru_cache


class WatermarkKey:
    """
    Manages secret key and derives watermark parameters.

    Supports two modes:
    - Fixed partition (v1): Same green set for all positions
    - Context-dependent partition (v2): Green set depends on previous token

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
        # Cache for context-dependent green token sets
        self._context_cache = {}

    # ─── Fixed Partition (v1 compatibility) ───

    def get_green_tokens(self) -> Set[int]:
        """
        Generate deterministic green token set from secret key.
        Uses SHA256 hash to partition vocabulary 50-50.
        This is the v1 fixed partition method.

        Returns:
            Set of token IDs classified as green (exactly 50%)
        """
        green_tokens = set()
        for token_id in range(self.vocab_size):
            hash_input = f"{self.secret_key}:token:{token_id}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
            if int(hash_value[:16], 16) % 2 == 0:
                green_tokens.add(token_id)
        return green_tokens

    def get_red_tokens(self) -> Set[int]:
        """
        Get the complement set of green tokens (red tokens).

        Returns:
            Set of token IDs classified as red
        """
        green = self.get_green_tokens()
        return set(range(self.vocab_size)) - green

    # ─── Context-Dependent Partition (v2) ───

    def get_green_tokens_context(self, prev_token_id: int) -> Set[int]:
        """
        Generate context-dependent green token set.
        The green/red partition changes based on the previous token,
        making the watermark signal much stronger for detection.

        Uses hash(secret_key, prev_token) as seed for a random
        permutation of the vocabulary. First half = green.

        Args:
            prev_token_id: Token ID of the previous token in the sequence

        Returns:
            Set of token IDs classified as green for this context
        """
        if prev_token_id in self._context_cache:
            return self._context_cache[prev_token_id]

        # Derive seed from secret key + previous token
        seed_input = f"{self.secret_key}:ctx:{prev_token_id}"
        seed = int(hashlib.sha256(seed_input.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed % (2**32))

        # Random permutation, first half is green
        perm = rng.permutation(self.vocab_size)
        green_tokens = set(perm[:self.vocab_size // 2].tolist())

        # Cache for reuse (common during detection)
        self._context_cache[prev_token_id] = green_tokens
        return green_tokens

    def is_green_token_context(self, token_id: int, prev_token_id: int) -> bool:
        """
        Check if a token is green given its predecessor.
        Efficient single-token check for detection.

        Args:
            token_id: Token to check
            prev_token_id: Previous token in sequence

        Returns:
            True if token_id is in the green set for this context
        """
        green_set = self.get_green_tokens_context(prev_token_id)
        return token_id in green_set

    def clear_context_cache(self):
        """Clear the context-dependent green token cache."""
        self._context_cache = {}

    # ─── Shared utilities ───

    def get_semantic_direction(self) -> np.ndarray:
        """
        Generate deterministic secret direction vector in embedding space.

        Returns:
            Normalized direction vector of shape (embedding_dim,)
        """
        seed = int(hashlib.sha256(self.secret_key.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed % (2**32))
        direction = rng.randn(self.embedding_dim)
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
        seed = int(hashlib.sha256(f"{self.secret_key}_burst".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed % (2**32))

        current_pos = 0
        while current_pos < max_length:
            jitter = rng.randint(-5, 6)
            start = max(0, current_pos + jitter)
            end = min(max_length, start + burst_length)
            positions.extend(range(start, end))
            current_pos += burst_interval

        return sorted(set(positions))

    def __repr__(self) -> str:
        """String representation of the key manager."""
        return (f"WatermarkKey(vocab_size={self.vocab_size}, "
                f"embedding_dim={self.embedding_dim})")
