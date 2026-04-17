"""Semantic search tool - embedding-based similarity matching."""

import math
import os
import pickle
import threading
from typing import Any, TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from arag.tools.base import BaseTool
from arag.tools.graph_utils import (
    build_local_subgraph,
    extract_mentions,
    run_local_graph_diffusion,
    select_query_mentions,
)
from arag.utils import get_env_float, get_env_int

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

class SemanticSearchTool(BaseTool):
    """Semantic search using embedding similarity."""

    _embedding_lock = threading.Lock()
    _MAX_TOP_K = 20

    def __init__(
        self,
        chunks_file: str,
        index_dir: str = "index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
    ):
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")

        self.index_dir = index_dir
        self.semantic_variant = os.environ.get("ARAG_SEMANTIC_VARIANT", "baseline").strip().lower()
        self.graph_ready = False

        self.embedding_model = SentenceTransformer(model_name, device=device)
        self._load_index()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "semantic_search"

    def _load_index(self):
        index_file = os.path.join(self.index_dir, "sentence_index.pkl")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file, "rb") as f:
            index_data = pickle.load(f)

        self.sentences = index_data["sentences"]
        self.embeddings = index_data["embeddings"]
        self.sentence_to_chunk = index_data["sentence_to_chunk"]
        self.chunks = index_data["chunks"]

        if self.semantic_variant == "baseline":
            return

        self.graph_data = dict()
        graph_index_file = os.path.join(self.index_dir, "graph_index.pkl")
        with open(graph_index_file, "rb") as f:
            self.graph_data = pickle.load(f)

        self.mention_df = self.graph_data.get("mention_df", [])
        self.mention_lookup = {t: i for i, t in enumerate(self.graph_data.get("mention_texts", []))}
        self.graph_ready = True

    def _compute_local_graph_scores(
        self,
        top_indices: np.ndarray,
        similarities: np.ndarray,
        query_mention_ids: set[int],
    ) -> tuple[dict[int, float], dict[str, int]]:
        """Run local graph reranking when the candidate pool has enough mention coverage."""
        mention_neighbors, sentence_edge_weights = build_local_subgraph(
            top_indices,
            query_mention_ids,
            mention_to_sentences=self.graph_data["mention_to_sentences"],
            mention_df=self.mention_df,
        )
        stats = {
            "active_mentions": len(mention_neighbors),
            "active_sentences": len(sentence_edge_weights),
        }
        if (
            len(mention_neighbors) < get_env_int("ARAG_LOCAL_GRAPH_MIN_COVERAGE", 2)
            or not sentence_edge_weights
        ):
            return {}, stats

        return (
            run_local_graph_diffusion(
                similarities,
                mention_neighbors,
                sentence_edge_weights,
                self.mention_df,
                seed_sentences=get_env_int("ARAG_GRAPH_SEED_SENTENCES", 5),
                ppr_steps=get_env_int("ARAG_GRAPH_PPR_STEPS", 6),
                restart_prob=get_env_float("ARAG_GRAPH_RESTART_PROB", 0.2),
            ),
            stats,
        )

    def _collect_chunk_matches(
        self,
        top_indices: np.ndarray,
        similarities: np.ndarray,
        query_mention_ids: set[int],
        graph_sentence_scores: dict[int, float],
    ) -> dict[str, list[dict[str, Any]]]:
        """Score candidate sentences and group them by chunk for final ranking."""
        alpha = get_env_float("ARAG_GRAPH_ALPHA", 1.0)
        beta = get_env_float("ARAG_GRAPH_BETA", 0.3)
        chunk_matches: dict[str, list[dict[str, Any]]] = {}

        for idx in top_indices:
            sentence = self.sentences[idx]
            chunk_id = self.sentence_to_chunk[idx]
            similarity = float(similarities[idx])
            score = similarity

            if self.semantic_variant == "mention_bonus" and query_mention_ids:
                mention_support = 0.0
                for mention_id in self.graph_data["sentence_mentions"].get(int(idx), []):
                    if mention_id not in query_mention_ids:
                        continue
                    df = self.mention_df[mention_id]
                    if df > 0:
                        mention_support += 1.0 / math.sqrt(df)
                score = alpha * similarity + beta * mention_support

            if self.semantic_variant == "local_graph" and graph_sentence_scores:
                score = alpha * similarity + beta * graph_sentence_scores.get(int(idx), 0.0)

            chunk_matches.setdefault(chunk_id, []).append(
                {
                    "sentence": sentence,
                    "similarity": similarity,
                    "score": score,
                }
            )

        return chunk_matches

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what information you're looking for",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant results to return (default: 5, max: 20)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self, context: "AgentContext", query: str, top_k: int = 5
    ) -> tuple[str, dict[str, Any]]:
        top_k = min(top_k, self._MAX_TOP_K)

        with self._embedding_lock:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]

        similarities = np.dot(self.embeddings, query_embedding)
        candidate_limit = top_k * 3

        query_mention_ids: set[int] = set()
        raw_query_mentions_count = 0
        graph_rerank_used = self.graph_ready and self.semantic_variant != "baseline"
        if graph_rerank_used:
            # Extract a small set of useful mentions from the query. Common single
            # words are filtered out so they do not dominate the local graph.
            raw_query_mentions = extract_mentions(query)
            raw_query_mentions_count = len(raw_query_mentions)
            query_mention_ids = select_query_mentions(
                raw_query_mentions,
                self.mention_lookup,
                self.mention_df
            )
            candidate_limit = max(get_env_int("ARAG_GRAPH_TOP_SENTENCES", 30), candidate_limit)

        top_indices = np.argsort(similarities)[::-1][:candidate_limit]
        graph_sentence_scores: dict[int, float] = {}
        graph_stats = {"active_mentions": 0, "active_sentences": 0}
        if self.semantic_variant == "local_graph" and query_mention_ids:
            graph_sentence_scores, graph_stats = self._compute_local_graph_scores(
                top_indices, similarities, query_mention_ids
            )

        # Sentence-level scores decide chunk ranking; final output still groups by chunk.
        chunk_sentences = self._collect_chunk_matches(
            top_indices,
            similarities,
            query_mention_ids,
            graph_sentence_scores,
        )
        chunk_scores = []
        for chunk_id, sentences in chunk_sentences.items():
            chunk_scores.append(
                (
                    chunk_id,
                    max(s["score"] for s in sentences),
                    max(s["similarity"] for s in sentences),
                    sentences,
                )
            )

        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]
        if not top_chunks:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts = []
        matched_sentences: list[str] = []
        for chunk_id, _, max_similarity, sentences in top_chunks:
            chunk_text = self.chunks[chunk_id]["text"]
            # Keep the snippet in source order so multiple matched sentences read naturally.
            sentences.sort(key=lambda item: chunk_text.find(item["sentence"]))
            snippet = "... " + " ... ".join(item["sentence"] for item in sentences) + " ..."
            result_parts.append(
                f"Chunk ID: {chunk_id} (Similarity: {max_similarity:.3f})\nMatched: {snippet}"
            )
            matched_sentences.extend(item["sentence"] for item in sentences)

        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(matched_sentences)))
            if matched_sentences
            else 0
        )
        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "chunks_found": len(top_chunks),
                "semantic_variant": self.semantic_variant,
                "candidate_sentences": candidate_limit,
                "query_mentions_count": len(query_mention_ids),
                "raw_query_mentions_count": raw_query_mentions_count,
                "graph_rerank_used": graph_rerank_used,
                "graph_stats": graph_stats,
            },
        )

        return "\n\n".join(result_parts), {
            "retrieved_tokens": retrieved_tokens,
            "chunks_found": len(top_chunks),
            "semantic_variant": self.semantic_variant,
        }
