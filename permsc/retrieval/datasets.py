__all__ = ['MSMarcoQueries', 'MSMarcoCollection', 'TRECQrels', 'TRECTop1000']

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from ..data import Item, RankingExample


class MSMarcoQueries:
    def __init__(self, queries_path: str):
        self.queries_path = Path(queries_path)
        self._queries: Dict[str, str] = {}
        self._load_queries()

    def _load_queries(self):
        with open(self.queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    query_id, query_text = parts
                    self._queries[query_id] = query_text

    def get_query(self, query_id: str) -> Optional[str]:
        return self._queries.get(query_id)

    def get_all_queries(self) -> Dict[str, str]:
        return self._queries.copy()

    def __len__(self) -> int:
        return len(self._queries)


class MSMarcoCollection:
    def __init__(self, collection_path: str):
        self.collection_path = Path(collection_path)
        self._passages: Dict[str, str] = {}
        self._index_path = self.collection_path.parent / f"{self.collection_path.stem}.index.pkl"

    def _load_collection(self):
        if self._index_path.exists():
            import pickle
            with open(self._index_path, 'rb') as f:
                self._passages = pickle.load(f)
            return

        print(f"Loading MS MARCO collection from {self.collection_path}...")
        print("This may take a while for large collections. Building index...")
        
        with open(self.collection_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"Processed {line_num} passages...")
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    passage_id, passage_text = parts
                    self._passages[passage_id] = passage_text

        import pickle
        with open(self._index_path, 'wb') as f:
            pickle.dump(self._passages, f)
        print(f"Index saved to {self._index_path}")

    def get_passage(self, passage_id: str) -> Optional[str]:
        if not self._passages:
            self._load_collection()
        return self._passages.get(passage_id)

    def get_all_passages(self) -> Dict[str, str]:
        if not self._passages:
            self._load_collection()
        return self._passages.copy()

    def __len__(self) -> int:
        if not self._passages:
            self._load_collection()
        return len(self._passages)


class TRECQrels:
    def __init__(self, qrels_path: str):
        self.qrels_path = Path(qrels_path)
        self._qrels: Dict[str, Dict[str, int]] = {}
        self._load_qrels()

    def _load_qrels(self):
        with open(self.qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    query_id, _, passage_id, relevance = parts
                    relevance = int(relevance)
                    if query_id not in self._qrels:
                        self._qrels[query_id] = {}
                    self._qrels[query_id][passage_id] = relevance

    def get_relevance(self, query_id: str, passage_id: str) -> int:
        return self._qrels.get(query_id, {}).get(passage_id, 0)

    def get_query_qrels(self, query_id: str) -> Dict[str, int]:
        return self._qrels.get(query_id, {}).copy()

    def get_all_qrels(self) -> Dict[str, Dict[str, int]]:
        return {qid: qrels.copy() for qid, qrels in self._qrels.items()}

    def __len__(self) -> int:
        return len(self._qrels)


class TRECTop1000:
    def __init__(self, top1000_path: str, collection: Optional[MSMarcoCollection] = None):
        self.top1000_path = Path(top1000_path)
        self.collection = collection
        self._results: Dict[str, List[tuple]] = {}
        self._load_top1000()

    def _load_top1000(self):
        print(f"Loading TREC top1000 results from {self.top1000_path}...")
        with open(self.top1000_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 4:
                    query_id, passage_id, query_text, passage_text = parts[0], parts[1], parts[2], parts[3]
                    
                    if query_id not in self._results:
                        self._results[query_id] = []
                    
                    self._results[query_id].append((passage_id, passage_text))

    def get_query_results(self, query_id: str) -> List[tuple]:
        return self._results.get(query_id, []).copy()

    def get_all_results(self) -> Dict[str, List[tuple]]:
        return {qid: results.copy() for qid, results in self._results.items()}

    def to_ranking_examples(self, queries: MSMarcoQueries) -> Dict[str, RankingExample]:
        examples = {}
        for query_id, results in self._results.items():
            query_text = queries.get_query(query_id)
            if query_text is None:
                continue
            
            hits = []
            for rank, (passage_id, passage_text) in enumerate(results):
                hits.append(Item(
                    content=passage_text,
                    id=passage_id,
                    score=0.0,
                    metadata={'rank': rank + 1}
                ))
            
            examples[query_id] = RankingExample(
                hits=hits,
                query=Item(content=query_text, id=query_id),
                metadata={'query_id': query_id}
            )
        
        return examples

    def __len__(self) -> int:
        return len(self._results)

