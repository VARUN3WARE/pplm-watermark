"""
Multi-signal watermark detector.

This module implements statistical detection of watermarks using
token bias analysis.

v1: Fixed green token partition - same green set for all positions.
v2: Context-dependent detection - green set depends on previous token,
    dramatically improving detection accuracy.
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from transformers import GPT2Tokenizer


class WatermarkDetector:
    """
    Watermark detector supporting both fixed and context-dependent modes.
    """

    def __init__(self, secret_key_manager, tokenizer: GPT2Tokenizer,
                 context_dependent: bool = True):
        """
        Initialize watermark detector.

        Args:
            secret_key_manager: WatermarkKey instance
            tokenizer: Tokenizer for text processing
            context_dependent: If True, use v2 context-dependent detection
        """
        self.key_manager = secret_key_manager
        self.tokenizer = tokenizer
        self.context_dependent = context_dependent
        
        if not context_dependent:
            # v1: Pre-compute fixed green token set
            self.green_tokens = secret_key_manager.get_green_tokens()
        
        self.semantic_direction = torch.tensor(
            secret_key_manager.get_semantic_direction()
        )

    def compute_token_bias_score(self, text: str) -> Tuple[float, Dict]:
        """
        Compute z-score for token bias signal.
        Automatically uses context-dependent or fixed mode.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (z_score, metadata_dict)
        """
        tokens = self.tokenizer.encode(text)
        n_tokens = len(tokens)

        if n_tokens < 20:
            return 0.0, {"error": "text too short", "n_tokens": n_tokens}

        if self.context_dependent:
            return self._compute_context_dependent_score(tokens)
        else:
            return self._compute_fixed_score(tokens)

    def _compute_fixed_score(self, tokens: List[int]) -> Tuple[float, Dict]:
        """
        Compute z-score using fixed green token partition (v1).

        Args:
            tokens: List of token IDs

        Returns:
            Tuple of (z_score, metadata_dict)
        """
        n_tokens = len(tokens)
        green_count = sum(1 for t in tokens if t in self.green_tokens)

        expected_green = n_tokens * 0.5
        variance = n_tokens * 0.5 * 0.5
        std_dev = np.sqrt(variance)

        z_score = (green_count - expected_green) / (std_dev + 1e-10)

        metadata = {
            "n_tokens": n_tokens,
            "green_count": green_count,
            "green_ratio": green_count / n_tokens,
            "expected_ratio": 0.5,
            "z_score": z_score,
            "mode": "fixed"
        }
        return z_score, metadata

    def _compute_context_dependent_score(self, tokens: List[int]) -> Tuple[float, Dict]:
        """
        Compute z-score using context-dependent green lists (v2).
        Each token is checked against the green set determined by
        its predecessor, making detection much more reliable.

        Args:
            tokens: List of token IDs

        Returns:
            Tuple of (z_score, metadata_dict)
        """
        n_tokens = len(tokens)
        
        # Check each token (starting from index 1) against green set
        # determined by the previous token
        green_count = 0
        scored_tokens = 0
        per_token_green = []
        
        for i in range(1, n_tokens):
            prev_token = tokens[i - 1]
            current_token = tokens[i]
            is_green = self.key_manager.is_green_token_context(current_token, prev_token)
            per_token_green.append(is_green)
            if is_green:
                green_count += 1
            scored_tokens += 1

        if scored_tokens < 10:
            return 0.0, {"error": "too few scored tokens", "scored_tokens": scored_tokens}

        # Under null hypothesis: each token has 50% chance of being green
        # (since green set is random permutation, independent of model choice)
        expected_green = scored_tokens * 0.5
        variance = scored_tokens * 0.5 * 0.5
        std_dev = np.sqrt(variance)

        z_score = (green_count - expected_green) / (std_dev + 1e-10)

        metadata = {
            "n_tokens": n_tokens,
            "scored_tokens": scored_tokens,
            "green_count": green_count,
            "green_ratio": green_count / scored_tokens,
            "expected_ratio": 0.5,
            "z_score": z_score,
            "mode": "context-dependent",
            "per_token_green": per_token_green
        }
        return z_score, metadata

    def compute_semantic_score(
        self,
        text: str,
        model
    ) -> Tuple[float, Dict]:
        """
        Compute semantic direction alignment score.

        Args:
            text: Input text to analyze
            model: Language model for embedding extraction

        Returns:
            Tuple of (alignment_score, metadata_dict)
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].mean(dim=1).squeeze()

        direction = self.semantic_direction.to(device)

        hidden_norm = hidden_states / (hidden_states.norm() + 1e-8)
        direction_norm = direction / (direction.norm() + 1e-8)
        direction_norm = direction_norm.to(hidden_norm.dtype)
        
        similarity = torch.dot(hidden_norm, direction_norm).item()

        metadata = {
            "semantic_similarity": similarity,
            "hidden_norm": hidden_states.norm().item()
        }
        return similarity, metadata

    def detect(
        self,
        text: str,
        model=None,
        token_weight: float = 1.0,
        semantic_weight: float = 0.0,
        threshold: float = 2.0
    ) -> Tuple[bool, float, Dict]:
        """
        Detect watermark in text.

        Args:
            text: Text to analyze
            model: Language model (required for semantic score)
            token_weight: Weight for token bias score (default 1.0)
            semantic_weight: Weight for semantic score (default 0.0)
            threshold: Detection threshold (z-score units)

        Returns:
            Tuple of (is_watermarked, combined_score, metadata)
        """
        token_z, token_meta = self.compute_token_bias_score(text)
        metadata = {"token": token_meta}

        if model is not None and semantic_weight > 0:
            semantic_score, semantic_meta = self.compute_semantic_score(text, model)
            metadata["semantic"] = semantic_meta
            semantic_z = semantic_score * 5
        else:
            semantic_z = 0
            semantic_weight = 0
            token_weight = 1.0

        combined_score = token_weight * token_z + semantic_weight * semantic_z
        is_watermarked = combined_score > threshold

        metadata["combined_score"] = combined_score
        metadata["threshold"] = threshold
        metadata["decision"] = is_watermarked

        return is_watermarked, combined_score, metadata

    def batch_detect(
        self,
        texts: list,
        model=None,
        threshold: float = 2.0
    ) -> list:
        """
        Detect watermark in multiple texts.

        Args:
            texts: List of text strings
            model: Language model for semantic scoring
            threshold: Detection threshold

        Returns:
            List of detection results (is_watermarked, score, metadata)
        """
        results = []
        for text in texts:
            result = self.detect(text, model=model, threshold=threshold)
            results.append(result)
        return results

    def __repr__(self) -> str:
        """String representation of the detector."""
        mode = "context-dependent" if self.context_dependent else "fixed"
        return f"WatermarkDetector(vocab_size={self.key_manager.vocab_size}, mode={mode})"
