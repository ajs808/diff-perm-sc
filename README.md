# MS MARCO Retrieval Pipeline with Permutation Self-Consistency

A full document retrieval pipeline for MS MARCO dataset that combines initial retrieval (BM25 or SPLADE++) with optional LLM reranking using permutation self-consistency.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pre-built Indexes

#### BM25 Index (Pyserini)

Download the pre-built MS MARCO passage index from Pyserini:

```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input <path-to-msmarco-collection> \
  --index indexes/msmarco-passage \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

Alternatively, download a pre-built index:
- Visit: https://github.com/castorini/pyserini#how-do-i-download-and-use-pre-built-indexes
- Look for "MS MARCO passage" index
- Download and extract to `indexes/msmarco-passage/`

#### SPLADE++ Index

SPLADE++ embeddings will be built on first run and cached. The model `naver/splade-cocondenser-ensembledistil` will be downloaded automatically from HuggingFace.

### 4. Configure API Keys (Optional)

For LLM reranking, create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

If no API key is provided, the pipeline will skip LLM reranking and only perform initial retrieval.

## Usage

See `experiments/msmarco_retrieval.ipynb` for a complete example of running the retrieval pipeline and evaluating on TREC DL19/20 datasets.

## Project Structure

- `permsc/retrieval/` - Retrieval pipeline components
  - `datasets.py` - MS MARCO and TREC dataset loaders
  - `retrievers.py` - BM25 and SPLADE++ retrievers
  - `pipeline.py` - Main retrieval pipeline orchestrator
  - `metrics.py` - NDCG and MRR evaluation metrics
- `permsc/llm/` - LLM reranking with permutation self-consistency
- `permsc/aggregator/` - Rank aggregation methods
- `experiments/` - Jupyter notebooks for experiments
- `data/` - MS MARCO and TREC datasets

## Evaluation

The pipeline evaluates retrieval performance using:
- **NDCG@k**: Normalized Discounted Cumulative Gain at rank k
- **MRR**: Mean Reciprocal Rank

Results are computed on TREC DL19 and DL20 query sets with official relevance judgments.

