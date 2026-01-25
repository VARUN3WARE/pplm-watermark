"""
Multi-signal watermark detector.

This module implements statistical detection of watermarks using
token bias and semantic direction signals.
"""

import numpy as np
import torch
from scipy import stats
from typing import Tuple, Dict, Optional
from transformers import GPT2Tokenizer


class WatermarkDetector:
    """
    Multi-signal watermark detector.
    """

    def __init__(self, secret_key_manager, tokenizer: GPT2Tokenizer):
        """
        Initialize watermark detector.

        Args:
            secret_key_manager: WatermarkKey instance
            tokenizer: Tokenizer for text processing
        """
        self.key_manager = secret_key_manager
        self.tokenizer = tokenizer
        self.green_tokens = secret_key_manager.get_green_tokens()
        self.semantic_direction = torch.tensor(
            secret_key_manager.get_semantic_direction()
        )

    def compute_token_bias_score(self, text: str) -> Tuple[float, Dict]:
        """
        Compute z-score for token bias signal.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (z_score, metadata_dict)
        """
        tokens = self.tokenizer.encode(text)
        n_tokens = len(tokens)

        if n_tokens < 20:
            return 0.0, {"error": "text too short", "n_tokens": n_tokens}

        # Count green tokens
        green_count = sum(1 for t in tokens if t in self.green_tokens)

        # Under null hypothesis: 50% green tokens
        # Under watermark: ~52% green tokens (subtle 2% bias)
        expected_green = n_tokens * 0.5
        variance = n_tokens * 0.5 * 0.5
        std_dev = np.sqrt(variance)

        # Compute z-score
        z_score = (green_count - expected_green) / (std_dev + 1e-10)

        metadata = {
            "n_tokens": n_tokens,
            "green_count": green_count,
            "green_ratio": green_count / n_tokens,
            "expected_ratio": 0.5,
            "z_score": z_score
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
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            # Use mean of last layer hidden states
            hidden_states = outputs.hidden_states[-1].mean(dim=1).squeeze()

        # Move direction to same device
        direction = self.semantic_direction.to(device)

        # Compute cosine similarity
        hidden_norm = hidden_states / (hidden_states.norm() + 1e-8)
        direction_norm = direction / (direction.norm() + 1e-8)

        # Ensure same dtype
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
        model = None,
        token_weight: float = 0.6,
        semantic_weight: float = 0.4,
        threshold: float = 2.0
    ) -> Tuple[bool, float, Dict]:
        """
        Combined watermark detection.

        Args:
            text: Text to analyze
            model: Language model (required for semantic score)
            token_weight: Weight for token bias score
            semantic_weight: Weight for semantic score
            threshold: Detection threshold (z-score units)

        Returns:
            Tuple of (is_watermarked, combined_score, metadata)
        """
        token_z, token_meta = self.compute_token_bias_score(text)

        metadata = {"token": token_meta}

        if model is not None:
            semantic_score, semantic_meta = self.compute_semantic_score(text, model)
            metadata["semantic"] = semantic_meta
            # Normalize semantic score to z-score scale
            semantic_z = semantic_score * 5  # Heuristic scaling
        else:
            semantic_z = 0
            semantic_weight = 0
            token_weight = 1.0

        # Compute combined score
        combined_score = token_weight * token_z + semantic_weight * semantic_z
        is_watermarked = combined_score > threshold

        metadata["combined_score"] = combined_score
        metadata["threshold"] = threshold
        metadata["decision"] = is_watermarked

        return is_watermarked, combined_score, metadata

    def batch_detect(
        self,
        texts: list,
        model = None,
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
        return f"WatermarkDetector(vocab_size={self.key_manager.vocab_size})"
