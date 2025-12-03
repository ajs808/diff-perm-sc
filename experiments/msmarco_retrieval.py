#!/usr/bin/env python3
"""
MS MARCO Retrieval Pipeline with Permutation Self-Consistency

This script demonstrates the full retrieval pipeline:
1. Initial retrieval using BM25 or SPLADE++
2. Optional LLM reranking with permutation self-consistency
3. Evaluation on TREC DL19/20 datasets using NDCG and MRR metrics
"""

import os
import sys
import argparse
from pathlib import Path
import glob

# Set JAVA_HOME for Pyserini (required for BM25 retrieval)
# Pyserini 0.39+ requires Java 21 (class file version 65.0)
# Always try to find and use Java 21, regardless of current JAVA_HOME setting
java_home_candidates = []

# First, try to find any installed Java 21, 17, or 11 via Homebrew dynamically
# Prefer Java 21 (required for Pyserini 0.39+), then 17, then 11
for version in ['21', '17', '11']:
    for pattern in [
        f'/usr/local/Cellar/openjdk@{version}/*/libexec/openjdk.jdk/Contents/Home',
        f'/opt/homebrew/opt/openjdk@{version}/libexec/openjdk.jdk/Contents/Home'
    ]:
        matches = glob.glob(pattern)
        if matches:
            java_home_candidates.extend(matches)

# Add fallback static paths (prefer Java 21)
java_home_candidates.extend([
    '/usr/local/Cellar/openjdk@21/21.0.9/libexec/openjdk.jdk/Contents/Home',
    '/usr/local/Cellar/openjdk@17/17.0.17/libexec/openjdk.jdk/Contents/Home',
    '/usr/local/Cellar/openjdk@11/11.0.29/libexec/openjdk.jdk/Contents/Home',
    '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home',
    '/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home',
    '/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home',
])

# Try to find and set Java 21 (or best available)
java_found = False
for candidate in java_home_candidates:
    if os.path.exists(candidate):
        # Verify it's actually Java 21 if possible
        try:
            java_version_output = os.popen(f"{candidate}/bin/java -version 2>&1").read()
            # Check if it's Java 21
            if '21' in java_version_output or candidate.find('openjdk@21') != -1:
                os.environ['JAVA_HOME'] = candidate
                os.environ['PATH'] = f"{candidate}/bin:{os.environ.get('PATH', '')}"
                print(f"Set JAVA_HOME to: {candidate}")
                java_found = True
                break
        except:
            pass

# If Java 21 not found, use the first available Java 11+ (but warn)
if not java_found:
    for candidate in java_home_candidates:
        if os.path.exists(candidate):
            os.environ['JAVA_HOME'] = candidate
            os.environ['PATH'] = f"{candidate}/bin:{os.environ.get('PATH', '')}"
            print(f"Set JAVA_HOME to: {candidate} (Note: Java 21 is recommended for Pyserini 0.39+)")
            java_found = True
            break

# Fallback: try to use java_home utility (prefer Java 21)
if not java_found:
    try:
        import subprocess
        for version in ['21', '17', '11+']:
            try:
                java_home = subprocess.check_output(['/usr/libexec/java_home', '-v', version]).decode().strip()
                os.environ['JAVA_HOME'] = java_home
                os.environ['PATH'] = f"{java_home}/bin:{os.environ.get('PATH', '')}"
                print(f"Set JAVA_HOME to: {java_home}")
                java_found = True
                break
            except:
                continue
    except:
        pass

if not java_found:
    print("Warning: Could not set JAVA_HOME automatically. Pyserini may not work.")
    print("Please ensure Java 21+ is installed and JAVA_HOME is set correctly.")
    print("Install with: brew install openjdk@21")

# Verify Java version
if 'JAVA_HOME' in os.environ:
    java_version = os.popen(f"{os.environ['JAVA_HOME']}/bin/java -version 2>&1").read()
    print(f"Java version: {java_version.split(chr(10))[0]}")

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

from permsc.retrieval import (
    MSMarcoQueries, MSMarcoCollection, TRECQrels,
    BM25Retriever, SpladeRetriever, RetrievalPipeline,
    evaluate_retrieval
)
from permsc.llm.openai_pool import OpenAIConfig

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MS MARCO Retrieval Pipeline with Permutation Self-Consistency'
    )
    parser.add_argument(
        '--retriever', 
        type=str, 
        choices=['bm25', 'splade'], 
        default='bm25',
        help='Retriever type: bm25 or splade (default: bm25)'
    )
    parser.add_argument(
        '--use-prebuilt-index',
        action='store_true',
        default=True,
        help='Use prebuilt BM25 index (downloads automatically)'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default=None,
        help='Path to local BM25 index (if not using prebuilt)'
    )
    parser.add_argument(
        '--prebuilt-index-name',
        type=str,
        default='msmarco-v1-passage',
        help='Name of prebuilt index (default: msmarco-v1-passage)'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        default=50,
        help='Maximum number of queries to evaluate (default: 50)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=1000,
        help='Number of initial retrieval results (default: 1000)'
    )
    parser.add_argument(
        '--rerank-depth',
        type=int,
        default=100,
        help='Number of top results to rerank with LLM (default: 100)'
    )
    parser.add_argument(
        '--num-permutations',
        type=int,
        default=5,
        help='Number of permutations for self-consistency (default: 5)'
    )
    parser.add_argument(
        '--aggregator',
        type=str,
        choices=['kemeny', 'rrf'],
        default='kemeny',
        help='Aggregation method: kemeny or rrf (default: kemeny)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results and plots (default: results)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing MS MARCO and TREC data (default: ../data)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Configuration
    print("=" * 60)
    print("MS MARCO Retrieval Pipeline")
    print("=" * 60)
    
    data_dir = Path(args.data_dir)
    msmarco_collection = data_dir / "msmarco/collection.tsv"
    trec_dl19_queries = data_dir / "trec-dl19/msmarco-test2019-queries.tsv"
    trec_dl19_qrels = data_dir / "trec-dl19/2019qrels-pass.txt"
    trec_dl20_queries = data_dir / "trec-dl20/msmarco-test2020-queries.tsv"
    trec_dl20_qrels = data_dir / "trec-dl20/2020qrels-pass.txt"
    
    print(f"\nData directory: {data_dir}")
    print(f"Retriever type: {args.retriever}")
    if args.retriever == 'bm25':
        if args.use_prebuilt_index:
            print(f"Using prebuilt index: {args.prebuilt_index_name} (will download if needed)")
        else:
            print(f"Using local index: {args.index_path}")
    
    # openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = ""
    print(f"OpenAI API key set: {bool(openai_api_key)}")
    
    # Load Datasets
    print("\n" + "=" * 60)
    print("Loading Datasets")
    print("=" * 60)
    
    print("Loading MS MARCO collection...")
    collection = MSMarcoCollection(str(msmarco_collection))
    print(f"Collection loaded: {len(collection)} passages")
    
    print("\nLoading TREC DL19 queries and qrels...")
    dl19_queries = MSMarcoQueries(str(trec_dl19_queries))
    dl19_qrels = TRECQrels(str(trec_dl19_qrels))
    print(f"DL19: {len(dl19_queries)} queries, {len(dl19_qrels)} queries with qrels")
    
    print("\nLoading TREC DL20 queries and qrels...")
    dl20_queries = MSMarcoQueries(str(trec_dl20_queries))
    dl20_qrels = TRECQrels(str(trec_dl20_qrels))
    print(f"DL20: {len(dl20_queries)} queries, {len(dl20_qrels)} queries with qrels")
    
    # Initialize Retrievers
    print("\n" + "=" * 60)
    print("Initializing Retriever")
    print("=" * 60)
    
    if args.retriever == "bm25":
        if args.use_prebuilt_index:
            print(f"Initializing BM25 retriever with prebuilt index: {args.prebuilt_index_name}")
            retriever = BM25Retriever(prebuilt_index=args.prebuilt_index_name)
        else:
            if not args.index_path:
                raise ValueError("--index-path must be provided when not using prebuilt index")
            print(f"Initializing BM25 retriever with local index: {args.index_path}")
            retriever = BM25Retriever(index_path=args.index_path)
    elif args.retriever == "splade":
        print("Initializing SPLADE++ retriever...")
        retriever = SpladeRetriever(str(msmarco_collection))
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever}")
    
    print("Retriever initialized successfully")
    
    # Setup LLM Reranking (Optional)
    print("\n" + "=" * 60)
    print("Setting up LLM Reranking")
    print("=" * 60)
    
    llm_config = None
    if openai_api_key:
        llm_config = OpenAIConfig(
            api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            api_type="openai"
        )
        print("LLM reranking enabled")
    else:
        print("No API key provided. LLM reranking disabled.")
    
    pipeline = RetrievalPipeline(
        retriever=retriever,
        collection=collection,
        llm_config=llm_config,
        num_permutations=args.num_permutations,
        aggregator=args.aggregator
    )
    
    # Run Retrieval and Evaluation
    print("\n" + "=" * 60)
    print("Running Retrieval and Evaluation")
    print("=" * 60)
    
    def run_evaluation(queries, qrels, dataset_name, max_queries=None):
        """Run retrieval pipeline and evaluate on a dataset."""
        query_ids = list(queries.get_all_queries().keys())
        if max_queries:
            query_ids = query_ids[:max_queries]
        
        results = {}
        
        print(f"\nRunning retrieval on {dataset_name} ({len(query_ids)} queries)...")
        for query_id in tqdm(query_ids, desc=f"{dataset_name} queries"):
            query_text = queries.get_query(query_id)
            if not query_text:
                continue
            
            ranking_example = pipeline.run(query_text, top_k=args.top_k, rerank_depth=args.rerank_depth)
            results[query_id] = ranking_example
        
        print(f"\nEvaluating {dataset_name}...")
        metrics = evaluate_retrieval(results, qrels.get_all_qrels(), k_values=[10, 100])
        
        return metrics, results
    
    metrics_dl19, results_dl19 = run_evaluation(dl19_queries, dl19_qrels, "DL19", max_queries=args.max_queries)
    metrics_dl20, results_dl20 = run_evaluation(dl20_queries, dl20_qrels, "DL20", max_queries=args.max_queries)
    
    # Results Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    results_df = pd.DataFrame({
        'Dataset': ['DL19', 'DL20'],
        'NDCG@10': [metrics_dl19['ndcg@10'], metrics_dl20['ndcg@10']],
        'NDCG@100': [metrics_dl19['ndcg@100'], metrics_dl20['ndcg@100']],
        'MRR': [metrics_dl19['mrr'], metrics_dl20['mrr']]
    })
    
    print("\nEvaluation Results:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    print(f"\nRetriever: {args.retriever.upper()}")
    print(f"LLM Reranking: {'Enabled' if llm_config else 'Disabled'}")
    if llm_config:
        print(f"Permutations: {pipeline.num_permutations}")
        print(f"Aggregator: {pipeline.aggregator_name}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Visualization
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['NDCG@10', 'NDCG@100', 'MRR']
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = results_df[metric].values
        ax.bar(results_df['Dataset'], values, color=['#3498db', '#e74c3c'])
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Dataset')
        ax.set_ylim(0, max(values) * 1.2)
        
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plot_file = output_dir / 'results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

