"""Helpers for lightweight mention extraction used by graph reranking."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

from arag.utils import get_env_int

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")

STOPWORDS = set(SPACY_STOP_WORDS)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def extract_mentions(text: str, min_chars: int = 4, max_ngram: int = 3) -> list[str]:
    """Extract conservative 1-3 gram mentions from English text."""
    tokens = TOKEN_RE.findall(text.lower())
    mentions: list[str] = []
    seen: set[str] = set()

    min_chars = min_chars or get_env_int("ARAG_MIN_MENTION_CHARS", 4)
    max_ngram = max_ngram or get_env_int("ARAG_MAX_MENTION_NGRAM", 3)

    for start in range(len(tokens)):
        for width in range(1, max_ngram + 1):
            end = start + width
            if end > len(tokens):
                break

            span = tokens[start:end]
            if all(token in STOPWORDS for token in span):
                continue
            if span[0] in STOPWORDS or span[-1] in STOPWORDS:
                continue

            mention = " ".join(span).strip()
            if len(mention) < min_chars or mention in seen:
                continue

            seen.add(mention)
            mentions.append(mention)

    return mentions


def normalize_values(values: dict[int, float]) -> dict[int, float]:
    """Scale values into [0, 1] while preserving key associations."""
    if not values:
        return {}

    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {key: 1.0 for key in values}

    scale = hi - lo
    return {key: (value - lo) / scale for key, value in values.items()}


def select_query_mentions(
    raw_mentions: list[str],
    mention_lookup: dict[str, int],
    mention_df: list[int],
) -> set[int]:
    """Pick the most informative query mentions for graph-based reranking."""
    if not raw_mentions:
        return set()

    single_mention_max_df = get_env_int("ARAG_SINGLE_MENTION_MAX_DF", 128)
    query_mention_limit = get_env_int("ARAG_QUERY_MENTION_LIMIT", 8)

    candidates: list[tuple[float, int, int, int]] = []
    for mention in raw_mentions:
        mention_id = mention_lookup.get(mention)
        if mention_id is None:
            continue

        token_count = len(mention.split())
        df = mention_df[mention_id]
        if token_count == 1 and df > single_mention_max_df:
            continue

        rarity = 1.0 / math.sqrt(df) if df > 0 else 0.0
        candidates.append((rarity + 0.15 * max(0, token_count - 1), token_count, -df, mention_id))

    candidates.sort(reverse=True)

    selected_ids: set[int] = set()
    for _, _, _, mention_id in candidates:
        selected_ids.add(mention_id)
        if len(selected_ids) >= query_mention_limit:
            break

    return selected_ids


def build_local_subgraph(
    top_indices: "NDArray[np.integer]",
    query_mention_ids: set[int],
    mention_to_sentences: dict[int, list[int]],
    mention_df: list[int],
) -> tuple[dict[int, list[int]], dict[int, dict[int, float]]]:
    """Build a mention-sentence subgraph restricted to the current retrieval pool."""
    candidate_sentence_ids = {int(idx) for idx in top_indices}
    mention_neighbors: dict[int, list[int]] = {}
    sentence_edge_weights: dict[int, dict[int, float]] = {}

    for mention_id in query_mention_ids:
        neighbors = [
            sentence_id
            for sentence_id in mention_to_sentences.get(mention_id, [])
            if sentence_id in candidate_sentence_ids
        ]
        if not neighbors:
            continue

        mention_neighbors[mention_id] = neighbors
        df = mention_df[mention_id]
        edge_weight = 1.0 / math.sqrt(df) if df > 0 else 0.0
        for sentence_id in neighbors:
            sentence_edge_weights.setdefault(sentence_id, {})[mention_id] = edge_weight

    return mention_neighbors, sentence_edge_weights


def run_local_graph_diffusion(
    similarities: "NDArray[np.floating]",
    mention_neighbors: dict[int, list[int]],
    sentence_edge_weights: dict[int, dict[int, float]],
    mention_df: list[int],
    *,
    seed_sentences: int,
    ppr_steps: int,
    restart_prob: float,
) -> dict[int, float]:
    """Diffuse scores on the local mention-sentence graph and return sentence scores."""
    ranked_sentence_ids = sorted(
        sentence_edge_weights,
        key=lambda sentence_id: float(similarities[sentence_id]),
        reverse=True,
    )
    sentence_seed = normalize_values(
        {
            sentence_id: max(float(similarities[sentence_id]), 0.0)
            for sentence_id in ranked_sentence_ids[:seed_sentences]
        }
    )
    mention_seed = normalize_values(
        {
            mention_id: 1.0 / math.sqrt(mention_df[mention_id])
            for mention_id in mention_neighbors
            if mention_df[mention_id] > 0
        }
    )

    sentence_scores = dict(sentence_seed)
    mention_scores = dict(mention_seed)
    zero_sentences = dict.fromkeys(sentence_edge_weights, 0.0)
    zero_mentions = dict.fromkeys(mention_neighbors, 0.0)

    for _ in range(ppr_steps):
        propagated_sentence = zero_sentences.copy()
        for mention_id, neighbors in mention_neighbors.items():
            weights = [sentence_edge_weights[sentence_id][mention_id] for sentence_id in neighbors]
            total_weight = sum(weights)
            if total_weight <= 0:
                continue

            for sentence_id, weight in zip(neighbors, weights, strict=False):
                propagated_sentence[sentence_id] += (
                    mention_scores.get(mention_id, 0.0) * weight / total_weight
                )

        next_sentence_scores = dict(zero_sentences)
        for sentence_id in sentence_edge_weights:
            next_sentence_scores[sentence_id] = (
                restart_prob * sentence_seed.get(sentence_id, 0.0)
                + (1.0 - restart_prob) * propagated_sentence.get(sentence_id, 0.0)
            )

        propagated_mentions = zero_mentions.copy()
        for sentence_id, mention_weights in sentence_edge_weights.items():
            total_weight = sum(mention_weights.values())
            if total_weight <= 0:
                continue

            for mention_id, weight in mention_weights.items():
                propagated_mentions[mention_id] += (
                    next_sentence_scores.get(sentence_id, 0.0) * weight / total_weight
                )

        next_mention_scores = dict(zero_mentions)
        for mention_id in mention_neighbors:
            next_mention_scores[mention_id] = (
                restart_prob * mention_seed.get(mention_id, 0.0)
                + (1.0 - restart_prob) * propagated_mentions.get(mention_id, 0.0)
            )

        sentence_scores = next_sentence_scores
        mention_scores = next_mention_scores

    return normalize_values(sentence_scores)
