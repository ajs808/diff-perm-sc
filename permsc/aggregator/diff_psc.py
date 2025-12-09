__all__ = ['DiffPSCAggregator']

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .base import RankAggregator


class DiffPSCAggregator(RankAggregator):
    """
    Differentiable Permutation Self-Consistency (Diff-PSC) aggregator.
    
    Implements the Diff-PSC algorithm which provides a fully differentiable
    relaxation of permutation self-consistency using NeuralSort.
    
    Args:
        temperature: NeuralSort temperature parameter (default: 1.0)
        use_final_neural_sort: Whether to apply final NeuralSort step (default: True)
        use_torch: Whether to use PyTorch for differentiable operations (default: True if available)
    """
    
    def __init__(self, 
                 temperature: float = 1.0,
                 use_final_neural_sort: bool = True,
                 use_torch: Optional[bool] = None):
        self.temperature = temperature
        self.use_final_neural_sort = use_final_neural_sort
        
        if use_torch is None:
            use_torch = TORCH_AVAILABLE
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        if self.use_torch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        """
        Aggregate preferences using Diff-PSC.
        
        Args:
            preferences: m x n preference matrix where each row is a ranking.
                        Values are item indices in ranked order.
        
        Returns:
            Aggregated preference as 1D array of length n.
        """
        m, n = preferences.shape
        
        # Convert preferences to scores (inverse ranks)
        # Higher rank position = higher score
        scores_list = []
        for i in range(m):
            pref = preferences[i]
            # Convert preference array to scores
            # Item at position 0 gets highest score, position n-1 gets lowest
            scores = np.zeros(n)
            for rank_pos, item_idx in enumerate(pref):
                if item_idx != -1:  # Skip missing items
                    scores[item_idx] = n - rank_pos
            scores_list.append(scores)
        
        scores_matrix = np.array(scores_list)  # m x n
        
        if self.use_torch:
            return self._aggregate_torch(scores_matrix, n)
        else:
            return self._aggregate_numpy(scores_matrix, n)
    
    def _aggregate_torch(self, scores_matrix: np.ndarray, n: int) -> np.ndarray:
        """PyTorch implementation for differentiable operations."""
        scores = torch.tensor(scores_matrix, dtype=torch.float32, device=self.device)  # m x n
        tau = self.temperature
        
        # Position weights: v_k = n + 1 - 2k
        v = torch.arange(1, n + 1, dtype=torch.float32, device=self.device)
        v = n + 1 - 2 * v  # n
        
        # Compute soft permutation matrices for each permutation
        all_pairwise_probs = []
        
        for i in range(scores.shape[0]):
            s = scores[i]  # n
            
            # Pairwise differences: A_{p,q} = |s_p - s_q|
            A = torch.abs(s.unsqueeze(0) - s.unsqueeze(1))  # n x n
            
            # Row sum for centering: b_p = sum_q A_{p,q}
            b = A.sum(dim=1)  # n
            
            # Score matrix: M_{p,k} = s_p * v_k - b_p
            M = s.unsqueeze(1) * v.unsqueeze(0) - b.unsqueeze(1)  # n x n
            
            # Soft permutation (row-wise softmax): P_{p,k} = exp(M_{p,k}/tau) / sum_t exp(M_{p,t}/tau)
            P = F.softmax(M / tau, dim=1)  # n x n
            
            # Compute pairwise ordering probabilities
            # p_{ab} = sum_{r=1}^n sum_{s=1}^{r-1} P_{a,r} * P_{b,s}
            # In 0-indexed: r goes 0..n-1, s goes 0..r-1 (which is range(r))
            pairwise_probs = torch.zeros(n, n, device=self.device)
            for a in range(n):
                for b in range(n):
                    if a != b:
                        prob = 0.0
                        for r in range(n):
                            for s in range(r):  # s from 0 to r-1 (positions before r)
                                prob += P[a, r] * P[b, s]
                        pairwise_probs[a, b] = prob
            
            all_pairwise_probs.append(pairwise_probs)
        
        # Aggregate across permutations: p_bar_{ab} = (1/m) * sum_i p^{(i)}_{ab}
        aggregated_pairwise = torch.stack(all_pairwise_probs).mean(dim=0)  # n x n
        
        # Compute aggregated item scores: s_tilde_j = sum_{k != j} p_bar_{jk}
        aggregated_scores = aggregated_pairwise.sum(dim=1)  # n
        
        if self.use_final_neural_sort:
            # Optional final soft central permutation via NeuralSort
            A_star = torch.abs(aggregated_scores.unsqueeze(0) - aggregated_scores.unsqueeze(1))
            b_star = A_star.sum(dim=1)
            M_star = aggregated_scores.unsqueeze(1) * v.unsqueeze(0) - b_star.unsqueeze(1)
            P_star = F.softmax(M_star / tau, dim=1)
            
            # Convert soft permutation to hard ranking (argmax along columns)
            # Each column k represents position k, so argmax gives which item is at position k
            ranking = P_star.argmax(dim=0).cpu().numpy()
        else:
            # Simple argsort on aggregated scores
            ranking = aggregated_scores.argsort(descending=True).cpu().numpy()
        
        return ranking
    
    def _aggregate_numpy(self, scores_matrix: np.ndarray, n: int) -> np.ndarray:
        """NumPy implementation for inference without PyTorch."""
        tau = self.temperature
        
        # Position weights: v_k = n + 1 - 2k
        v = np.arange(1, n + 1, dtype=np.float32)
        v = n + 1 - 2 * v  # n
        
        # Compute soft permutation matrices for each permutation
        all_pairwise_probs = []
        
        for i in range(scores_matrix.shape[0]):
            s = scores_matrix[i]  # n
            
            # Pairwise differences: A_{p,q} = |s_p - s_q|
            A = np.abs(s[:, np.newaxis] - s[np.newaxis, :])  # n x n
            
            # Row sum for centering: b_p = sum_q A_{p,q}
            b = A.sum(axis=1)  # n
            
            # Score matrix: M_{p,k} = s_p * v_k - b_p
            M = s[:, np.newaxis] * v[np.newaxis, :] - b[:, np.newaxis]  # n x n
            
            # Soft permutation (row-wise softmax): P_{p,k} = exp(M_{p,k}/tau) / sum_t exp(M_{p,t}/tau)
            M_scaled = M / tau
            M_scaled = M_scaled - M_scaled.max(axis=1, keepdims=True)  # Numerical stability
            exp_M = np.exp(M_scaled)
            P = exp_M / exp_M.sum(axis=1, keepdims=True)  # n x n
            
            # Compute pairwise ordering probabilities
            # p_{ab} = sum_{r=1}^n sum_{s=1}^{r-1} P_{a,r} * P_{b,s}
            # In 0-indexed: r goes 0..n-1, s goes 0..r-1 (which is range(r))
            pairwise_probs = np.zeros((n, n))
            for a in range(n):
                for b in range(n):
                    if a != b:
                        prob = 0.0
                        for r in range(n):
                            for s in range(r):  # s from 0 to r-1 (positions before r)
                                prob += P[a, r] * P[b, s]
                        pairwise_probs[a, b] = prob
            
            all_pairwise_probs.append(pairwise_probs)
        
        # Aggregate across permutations: p_bar_{ab} = (1/m) * sum_i p^{(i)}_{ab}
        aggregated_pairwise = np.mean(all_pairwise_probs, axis=0)  # n x n
        
        # Compute aggregated item scores: s_tilde_j = sum_{k != j} p_bar_{jk}
        aggregated_scores = aggregated_pairwise.sum(axis=1)  # n
        
        if self.use_final_neural_sort:
            # Optional final soft central permutation via NeuralSort
            A_star = np.abs(aggregated_scores[:, np.newaxis] - aggregated_scores[np.newaxis, :])
            b_star = A_star.sum(axis=1)
            M_star = aggregated_scores[:, np.newaxis] * v[np.newaxis, :] - b_star[:, np.newaxis]
            M_star_scaled = M_star / tau
            M_star_scaled = M_star_scaled - M_star_scaled.max(axis=1, keepdims=True)
            exp_M_star = np.exp(M_star_scaled)
            P_star = exp_M_star / exp_M_star.sum(axis=1, keepdims=True)
            
            # Convert soft permutation to hard ranking (argmax along columns)
            ranking = P_star.argmax(axis=0)
        else:
            # Simple argsort on aggregated scores
            ranking = np.argsort(aggregated_scores)[::-1]
        
        return ranking

