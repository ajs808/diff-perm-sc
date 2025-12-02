__all__ = ['BaseRetriever', 'BM25Retriever', 'SpladeRetriever']

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Retrieve top_k passages for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (passage_id, score) tuples, sorted by score descending
        """
        pass


class BM25Retriever(BaseRetriever):
    def __init__(self, index_path: str = None, prebuilt_index: str = None):
        """
        Initialize BM25 retriever.
        
        Args:
            index_path: Path to local index directory (optional if prebuilt_index is provided)
            prebuilt_index: Name of prebuilt index (e.g., 'msmarco-v1-passage'). 
                          If provided, will download and cache automatically.
        """
        if prebuilt_index:
            self.index_path = None
            self.prebuilt_index = prebuilt_index
            self._use_prebuilt = True
        elif index_path:
            self.index_path = Path(index_path)
            self.prebuilt_index = None
            self._use_prebuilt = False
            self._check_index()
        else:
            raise ValueError("Either index_path or prebuilt_index must be provided")
        
        self._searcher = None

    def _check_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.index_path}\n"
                f"Please download a pre-built index or build one using:\n"
                f"python -m pyserini.index.lucene \\\n"
                f"  --collection JsonCollection \\\n"
                f"  --input <path-to-msmarco-collection> \\\n"
                f"  --index {self.index_path} \\\n"
                f"  --generator DefaultLuceneDocumentGenerator \\\n"
                f"  --threads 1 \\\n"
                f"  --storePositions --storeDocvectors --storeRaw\n\n"
                f"Or use prebuilt_index='msmarco-v1-passage' to download automatically, "
                f"or download from: https://github.com/castorini/pyserini#how-do-i-download-and-use-pre-built-indexes"
            )

    def _get_searcher(self):
        if self._searcher is None:
            try:
                from pyserini.search import LuceneSearcher
                if self._use_prebuilt:
                    print(f"Loading prebuilt index '{self.prebuilt_index}' (will download if not cached)...")
                    self._searcher = LuceneSearcher.from_prebuilt_index(self.prebuilt_index)
                    print(f"Index loaded successfully")
                else:
                    self._searcher = LuceneSearcher(str(self.index_path))
            except ImportError:
                raise ImportError(
                    "pyserini is required for BM25 retrieval. Install with: pip install pyserini"
                )
        return self._searcher

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        searcher = self._get_searcher()
        hits = searcher.search(query, k=top_k)
        
        results = []
        for hit in hits:
            passage_id = hit.docid
            score = hit.score
            results.append((passage_id, float(score)))
        
        return results


class SpladeRetriever(BaseRetriever):
    def __init__(self, collection_path: str, model_name: str = 'naver/splade-cocondenser-ensembledistil', 
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        self.collection_path = Path(collection_path)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'splade_embeddings'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self._model = None
        self._passage_embeddings = None
        self._passage_ids = None
        self._embedding_cache_path = self.cache_dir / f"{self.collection_path.stem}_embeddings.npy"
        self._ids_cache_path = self.cache_dir / f"{self.collection_path.stem}_ids.npy"

    def _load_model(self):
        if self._model is None:
            print(f"Loading SPLADE++ model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _load_collection(self):
        if self._passage_ids is not None:
            return
        
        print(f"Loading passage collection from {self.collection_path}...")
        passage_ids = []
        passages = []
        
        with open(self.collection_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 100000 == 0:
                    print(f"Loaded {line_num} passages...")
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    passage_id, passage_text = parts
                    passage_ids.append(passage_id)
                    passages.append(passage_text)
        
        self._passage_ids = np.array(passage_ids)
        self._passages = passages
        print(f"Loaded {len(passage_ids)} passages")

    def _build_embeddings(self):
        if self._passage_embeddings is not None:
            return
        
        if self._embedding_cache_path.exists() and self._ids_cache_path.exists():
            print(f"Loading cached embeddings from {self._embedding_cache_path}")
            self._passage_embeddings = np.load(self._embedding_cache_path)
            self._passage_ids = np.load(self._ids_cache_path, allow_pickle=True)
            print(f"Loaded {len(self._passage_ids)} cached embeddings")
            return
        
        self._load_collection()
        model = self._load_model()
        
        print("Computing SPLADE++ embeddings for all passages...")
        print("This may take a while. Embeddings will be cached for future use.")
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(self._passages), batch_size):
            batch = self._passages[i:i + batch_size]
            if (i // batch_size) % 100 == 0:
                print(f"Processed {i}/{len(self._passages)} passages...")
            
            embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(embeddings)
        
        self._passage_embeddings = np.vstack(all_embeddings)
        
        print(f"Saving embeddings to {self._embedding_cache_path}")
        np.save(self._embedding_cache_path, self._passage_embeddings)
        np.save(self._ids_cache_path, self._passage_ids)
        print("Embeddings cached successfully")

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        self._build_embeddings()
        model = self._load_model()
        
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        
        scores = np.dot(self._passage_embeddings, query_embedding)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            passage_id = self._passage_ids[idx]
            score = float(scores[idx])
            results.append((passage_id, score))
        
        return results

