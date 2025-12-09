# Diff-PSC

Project members: Arul Saxena

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate 
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets and Indexes

#### MS MARCO Collection

Download the MS MARCO passage collection and place it at:
```
data/msmarco/collection.tsv
```

Format: tab-separated file with `passage_id\tpassage_text` per line.

#### TREC DL19/20 Datasets

Download TREC Deep Learning 2019/2020 datasets and place them at:

**TREC DL19:**
- `data/trec-dl19/msmarco-test2019-queries.tsv`
- `data/trec-dl19/2019qrels-pass.txt`

**TREC DL20:**
- `data/trec-dl20/msmarco-test2020-queries.tsv`
- `data/trec-dl20/2020qrels-pass.txt`

#### BM25 Index

**Option 1: Use prebuilt index (recommended)**
- No download needed. The pipeline will automatically download and cache the index on first use.
- Specify with `--use-prebuilt-index` (default).

**Option 2: Use local index**
- Download a prebuilt index from [Pyserini](https://github.com/castorini/pyserini#how-do-i-download-and-use-pre-built-indexes)
- Extract to a directory (e.g., `indexes/msmarco-passage/`)
- Specify with `--index-path <path-to-index-directory>`

**Option 3: Build index from collection**
```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/msmarco/collection.tsv \
  --index indexes/msmarco-passage \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

#### SPLADE++ Index

No setup needed. SPLADE++ embeddings are built on first run and cached automatically.

### 4. Configure API Keys (Optional)

Set `OPENAI_API_KEY` environment variable for LLM reranking. If not set, the pipeline will skip LLM reranking and only perform initial retrieval.

## Usage

Run the retrieval pipeline:

```bash
python experiments/msmarco_retrieval.py \
  --retriever bm25 \
  --aggregator kemeny \
  --data-dir data \
  --max-queries 50
```

Available aggregators: `kemeny`, `rrf`, `diff_psc`, `tideman`

See `experiments/msmarco_retrieval.ipynb` for a complete example.
