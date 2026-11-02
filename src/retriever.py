"""
Multi-Engine Multilingual Information Retrieval Module
Supports:
1. Lexical Retrieval: Okapi BM25 with cross-lingual tokenization & n-grams
2. Dense Semantic Retrieval: Multilingual Transformer (MiniLM-L12-v2)
3. Hybrid Retrieval: Reciprocal Rank Fusion (RRF) + Normalized Weighted Score Combination
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set offline flags before importing HuggingFace / Transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from rank_bm25 import BM25Okapi
import torch
from sentence_transformers import SentenceTransformer

from src.normalizer import MultilingualNormalizer


class MultilingualRetriever:
    """
    Unified Multilingual Information Retrieval System for Pakistan Public Service Documents.
    """

    def __init__(
        self,
        knowledge_base_path: str = 'data/knowledge_base.json',
        model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        embeddings_cache_path: str = 'data/passage_embeddings.npz'
    ):
        self.knowledge_base_path = knowledge_base_path
        self.model_name = model_name
        self.embeddings_cache_path = embeddings_cache_path
        self.normalizer = MultilingualNormalizer()

        # Load Knowledge Base
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.passages = json.load(f)

        self.passage_ids = [p['passage_id'] for p in self.passages]
        self.num_passages = len(self.passages)

        # 1. Initialize Lexical BM25 Index
        self._build_lexical_index()

        # 2. Initialize Dense Semantic Model & Embeddings
        self._init_dense_model()

    def _build_lexical_index(self):
        """Builds BM25 index over both English and Urdu content and topic metadata."""
        self.corpus_tokens = []
        for p in self.passages:
            # Combine English and Urdu text plus topic for comprehensive lexical match
            doc_text = f"{p.get('topic', '')} {p.get('gold_evidence_en', '')} {p.get('gold_evidence_ur', '')}"
            tokens = self.normalizer.tokenize_for_retrieval(doc_text)
            self.corpus_tokens.append(tokens)

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _init_dense_model(self):
        """Loads dense multilingual embedding model and pre-computes / loads cached embeddings."""
        try:
            self.dense_model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.dense_model = SentenceTransformer(self.model_name)

        if os.path.exists(self.embeddings_cache_path):
            data = np.load(self.embeddings_cache_path)
            self.passage_embeddings_en = data['en']
            self.passage_embeddings_ur = data['ur']
            self.passage_embeddings_joint = data['joint']
        else:
            self._compute_and_cache_embeddings()

    def _compute_and_cache_embeddings(self):
        """Computes embeddings for all passages and saves to disk cache."""
        en_texts = [f"{p.get('topic', '')}: {p.get('gold_evidence_en', '')}" for p in self.passages]
        ur_texts = [f"{p.get('topic', '')}: {p.get('gold_evidence_ur', '')}" for p in self.passages]
        joint_texts = [f"{p.get('topic', '')}: {p.get('gold_evidence_en', '')} | {p.get('gold_evidence_ur', '')}" for p in self.passages]

        self.passage_embeddings_en = self.dense_model.encode(en_texts, normalize_embeddings=True, show_progress_bar=False)
        self.passage_embeddings_ur = self.dense_model.encode(ur_texts, normalize_embeddings=True, show_progress_bar=False)
        self.passage_embeddings_joint = self.dense_model.encode(joint_texts, normalize_embeddings=True, show_progress_bar=False)

        os.makedirs(os.path.dirname(self.embeddings_cache_path) or '.', exist_ok=True)
        np.savez(
            self.embeddings_cache_path,
            en=self.passage_embeddings_en,
            ur=self.passage_embeddings_ur,
            joint=self.passage_embeddings_joint
        )

    def retrieve_lexical(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k passages using BM25.
        """
        t0 = time.perf_counter()
        query_norm, lang = self.normalizer.normalize(query)
        query_tokens = self.normalizer.tokenize_for_retrieval(query, lang=lang)

        scores = self.bm25.get_scores(query_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            p = self.passages[idx].copy()
            p['score'] = float(scores[idx])
            p['rank'] = rank
            p['retrieval_method'] = 'bm25_lexical'
            p['query_lang'] = lang
            p['elapsed_ms'] = elapsed_ms
            results.append(p)
        return results

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k passages using Dense Multilingual Embeddings.
        Uses dual-script cosine similarity (maximum of EN, UR, and Joint alignment).
        """
        t0 = time.perf_counter()
        query_norm, lang = self.normalizer.normalize(query)
        query_embedding = self.dense_model.encode(query, normalize_embeddings=True, show_progress_bar=False)

        # Dot product with normalized embeddings is equivalent to Cosine Similarity
        sim_en = np.dot(self.passage_embeddings_en, query_embedding)
        sim_ur = np.dot(self.passage_embeddings_ur, query_embedding)
        sim_joint = np.dot(self.passage_embeddings_joint, query_embedding)

        # Cross-lingual alignment: take best alignment score across representations
        dense_scores = np.maximum(np.maximum(sim_en, sim_ur), sim_joint)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(dense_scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            p = self.passages[idx].copy()
            p['score'] = float(dense_scores[idx])
            p['rank'] = rank
            p['retrieval_method'] = 'dense_semantic'
            p['query_lang'] = lang
            p['elapsed_ms'] = elapsed_ms
            results.append(p)
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k passages using Reciprocal Rank Fusion (RRF) and Weighted Score Combination.
        """
        t0 = time.perf_counter()
        query_norm, lang = self.normalizer.normalize(query)

        # 1. Lexical scores
        query_tokens = self.normalizer.tokenize_for_retrieval(query, lang=lang)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)

        # 2. Dense scores
        query_embedding = self.dense_model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        sim_en = np.dot(self.passage_embeddings_en, query_embedding)
        sim_ur = np.dot(self.passage_embeddings_ur, query_embedding)
        sim_joint = np.dot(self.passage_embeddings_joint, query_embedding)
        dense_scores = np.maximum(np.maximum(sim_en, sim_ur), sim_joint)

        # Compute ranks for RRF
        bm25_order = np.argsort(bm25_scores)[::-1]
        dense_order = np.argsort(dense_scores)[::-1]

        bm25_ranks = {idx: r + 1 for r, idx in enumerate(bm25_order)}
        dense_ranks = {idx: r + 1 for r, idx in enumerate(dense_order)}

        # Min-max normalization for linear combination
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = np.clip(bm25_scores / bm25_max, 0, 1)

        dense_min = dense_scores.min()
        dense_max = dense_scores.max()
        dense_norm = (dense_scores - dense_min) / (dense_max - dense_min + 1e-9) if dense_max > dense_min else dense_scores

        # Combine using RRF + Score weighting
        hybrid_scores = np.zeros(self.num_passages)
        for i in range(self.num_passages):
            rrf_score = (lexical_weight / (rrf_k + bm25_ranks[i])) + (dense_weight / (rrf_k + dense_ranks[i]))
            linear_score = (lexical_weight * bm25_norm[i]) + (dense_weight * dense_norm[i])
            hybrid_scores[i] = 0.5 * (rrf_score * 100) + 0.5 * linear_score

        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            p = self.passages[idx].copy()
            p['score'] = float(hybrid_scores[idx])
            p['dense_score'] = float(dense_scores[idx])
            p['bm25_score'] = float(bm25_scores[idx])
            p['rank'] = rank
            p['retrieval_method'] = 'hybrid_rrf'
            p['query_lang'] = lang
            p['elapsed_ms'] = elapsed_ms
            results.append(p)
        return results

    def retrieve(self, query: str, method: str = 'hybrid', top_k: int = 5) -> List[Dict[str, Any]]:
        """Dispatcher method for retrieval."""
        if method == 'bm25':
            return self.retrieve_lexical(query, top_k=top_k)
        elif method == 'dense':
            return self.retrieve_dense(query, top_k=top_k)
        else:
            return self.retrieve_hybrid(query, top_k=top_k)
