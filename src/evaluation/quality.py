"""
Quality evaluation for watermarked text.

Measures perplexity and other quality metrics to ensure
watermark is imperceptible.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List


class QualityEvaluator:
    """
    Evaluates quality of watermarked text.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize quality evaluator.
        
        Args:
            model_name: HuggingFace model name
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of text under the language model.
        
        Lower perplexity = more natural text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        if tokens.size(1) < 2:
            return float('inf')
        
        with torch.no_grad():
            outputs = self.model(tokens, labels=tokens)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def compute_perplexity_increase(self, 
                                   watermarked_text: str, 
                                   clean_text: str) -> Dict[str, float]:
        """
        Compare perplexity between watermarked and clean text.
        
        Args:
            watermarked_text: Watermarked text
            clean_text: Clean baseline text
            
        Returns:
            Dictionary with perplexity metrics
        """
        wm_ppl = self.compute_perplexity(watermarked_text)
        clean_ppl = self.compute_perplexity(clean_text)
        
        absolute_increase = wm_ppl - clean_ppl
        relative_increase = (wm_ppl - clean_ppl) / clean_ppl if clean_ppl > 0 else float('inf')
        
        return {
            'watermarked_perplexity': wm_ppl,
            'clean_perplexity': clean_ppl,
            'absolute_increase': absolute_increase,
            'relative_increase': relative_increase,
            'percentage_increase': relative_increase * 100
        }
    
    def evaluate_batch(self, 
                      watermarked_texts: List[str],
                      clean_texts: List[str]) -> Dict[str, float]:
        """
        Evaluate quality over multiple samples.
        
        Args:
            watermarked_texts: List of watermarked texts
            clean_texts: List of clean texts
            
        Returns:
            Average quality metrics
        """
        import numpy as np
        
        wm_ppls = [self.compute_perplexity(text) for text in watermarked_texts]
        clean_ppls = [self.compute_perplexity(text) for text in clean_texts]
        
        avg_wm_ppl = np.mean(wm_ppls)
        avg_clean_ppl = np.mean(clean_ppls)
        
        relative_increase = (avg_wm_ppl - avg_clean_ppl) / avg_clean_ppl if avg_clean_ppl > 0 else float('inf')
        
        return {
            'avg_watermarked_perplexity': avg_wm_ppl,
            'std_watermarked_perplexity': np.std(wm_ppls),
            'avg_clean_perplexity': avg_clean_ppl,
            'std_clean_perplexity': np.std(clean_ppls),
            'avg_absolute_increase': avg_wm_ppl - avg_clean_ppl,
            'avg_relative_increase': relative_increase,
            'avg_percentage_increase': relative_increase * 100,
            'quality_preserved': relative_increase < 0.1  # <10% increase = good
        }
