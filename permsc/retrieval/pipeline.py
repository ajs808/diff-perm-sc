__all__ = ['RetrievalPipeline']

from copy import deepcopy
from typing import Optional, List
import numpy as np

from ..data import RankingExample, Item
from ..llm.prompt_builder import RelevanceRankingPromptBuilder
from ..llm.prompt_pipeline import OpenAIPromptPipeline
from ..llm.openai_pool import ChatCompletionPool, OpenAIConfig
from ..aggregator import KemenyOptimalAggregator, RRFRankAggregator, TidemanRankedPairsAggregator
try:
    from ..aggregator import DiffPSCAggregator
except ImportError:
    DiffPSCAggregator = None
from .retrievers import BaseRetriever
from .datasets import MSMarcoCollection


class RetrievalPipeline:
    def __init__(self, 
                 retriever: BaseRetriever,
                 collection: MSMarcoCollection,
                 llm_config: Optional[OpenAIConfig] = None,
                 num_permutations: int = 5,
                 aggregator: str = 'kemeny'):
        """
        Initialize retrieval pipeline.
        
        Args:
            retriever: BM25 or SPLADE++ retriever
            collection: MS MARCO collection for retrieving passage text
            llm_config: Optional OpenAI config for LLM reranking. If None or no API key, reranking is skipped.
            num_permutations: Number of permutations for self-consistency (default: 5)
            aggregator: Aggregation method ('kemeny', 'rrf', 'diff_psc', or 'tideman', default: 'kemeny')
        """
        self.retriever = retriever
        self.collection = collection
        self.num_permutations = num_permutations
        self.aggregator_name = aggregator
        
        self._llm_enabled = False
        self._prompt_pipeline = None
        self._aggregator = None
        
        if llm_config and llm_config.api_key:
            api_key = llm_config.api_key
            if not api_key or api_key.strip() == '':
                print("Warning: LLM API key is empty. LLM reranking will be disabled.")
            else:
                self._setup_llm_reranking(llm_config)
        else:
            print("No LLM config provided or API key missing. LLM reranking will be disabled.")
    
    def _setup_llm_reranking(self, llm_config: OpenAIConfig):
        try:
            pool = ChatCompletionPool([llm_config])
            prompt_builder = RelevanceRankingPromptBuilder()
            self._prompt_pipeline = OpenAIPromptPipeline(prompt_builder, pool)
            
            if self.aggregator_name == 'kemeny':
                self._aggregator = KemenyOptimalAggregator()
            elif self.aggregator_name == 'rrf':
                self._aggregator = RRFRankAggregator()
            elif self.aggregator_name == 'diff_psc':
                if DiffPSCAggregator is None:
                    raise ValueError("DiffPSCAggregator is not available. Ensure all dependencies are installed.")
                self._aggregator = DiffPSCAggregator()
            elif self.aggregator_name == 'tideman':
                self._aggregator = TidemanRankedPairsAggregator()
            else:
                raise ValueError(f"Unknown aggregator: {self.aggregator_name}")
            
            self._llm_enabled = True
            print(f"LLM reranking enabled with {self.aggregator_name} aggregation")
        except Exception as e:
            print(f"Warning: Failed to setup LLM reranking: {e}. Continuing without reranking.")
            self._llm_enabled = False
    
    def run(self, query: str, top_k: int, rerank_depth: int = 100) -> RankingExample:
        """
        Run retrieval pipeline for a query.
        
        Args:
            query: Query text
            top_k: Number of initial retrieval results
            rerank_depth: Number of top results to rerank with LLM (only if LLM enabled)
            
        Returns:
            RankingExample with query and ranked hits
        """
        retrieval_results = self.retriever.retrieve(query, top_k)
        
        hits = []
        for passage_id, score in retrieval_results:
            passage_text = self.collection.get_passage(passage_id)
            if passage_text is None:
                continue
            
            hits.append(Item(
                content=passage_text,
                id=passage_id,
                score=score,
                metadata={'rank': len(hits) + 1}
            ))
        
        ranking_example = RankingExample(
            hits=hits,
            query=Item(content=query),
            metadata={}
        )
        
        if self._llm_enabled and len(hits) > 0:
            ranking_example = self._rerank_with_llm(ranking_example, rerank_depth)
        
        return ranking_example
    
    def _rerank_with_llm(self, ranking_example: RankingExample, rerank_depth: int) -> RankingExample:
        """
        Rerank top results using LLM with permutation self-consistency.
        """
        if len(ranking_example.hits) <= rerank_depth:
            to_rerank = deepcopy(ranking_example)
            remaining = RankingExample(hits=[], query=ranking_example.query, metadata={})
        else:
            to_rerank = RankingExample(
                hits=deepcopy(ranking_example.hits[:rerank_depth]),
                query=deepcopy(ranking_example.query),
                metadata={}
            )
            remaining = RankingExample(
                hits=deepcopy(ranking_example.hits[rerank_depth:]),
                query=deepcopy(ranking_example.query),
                metadata={}
            )
        
        all_preferences = []
        
        for perm_idx in range(self.num_permutations):
            permuted_example = deepcopy(to_rerank)
            permuted_example.randomize_order()
            
            preferences = self._prompt_pipeline.run(permuted_example)
            
            original_preferences = permuted_example.permuted_preferences_to_original_order(preferences)
            all_preferences.append(original_preferences)
        
        if not all_preferences:
            return ranking_example
        
        preferences_matrix = np.array(all_preferences)
        aggregated_preferences = self._aggregator.aggregate(preferences_matrix)
        
        reranked_hits = [to_rerank.hits[i] for i in aggregated_preferences if i != -1]
        
        for rank, hit in enumerate(reranked_hits):
            hit.metadata['rank'] = rank + 1
            hit.score = 1.0 / (rank + 1)
        
        final_hits = reranked_hits + remaining.hits
        for rank, hit in enumerate(final_hits):
            hit.metadata['rank'] = rank + 1
        
        return RankingExample(
            hits=final_hits,
            query=ranking_example.query,
            metadata=ranking_example.metadata
        )

