#!/usr/bin/env python3
"""
Build sentence-level embedding index for semantic search.

Usage:
    python scripts/build_index.py \
        --chunks data/chunks.json \
        --output data/index \
        --model sentence-transformers/all-MiniLM-L6-v2
"""

import json
import re
import pickle
import argparse
from pathlib import Path
from typing import Any

from rich.progress import track
from arag.tools.graph_utils import extract_mentions
from arag.utils import get_env_float, get_env_int


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def load_chunks(chunks_file: str) -> list[dict[str, Any]]:
    """Load chunks from file."""
    with open(chunks_file, encoding="utf-8") as f:
        data = json.load(f)

    if data and isinstance(data[0], dict):
        return data

    chunks = []
    for item in data:
        if isinstance(item, str):
            parts = item.split(":", 1)
            if len(parts) == 2:
                chunks.append({"id": parts[0], "text": parts[1]})
    return chunks


def build_graph_index(
    sentences: list[str],
    sentence_to_chunk: list[str],
    min_mention_chars: int,
    max_mention_df_ratio: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build a lightweight mention/sentence/chunk graph index."""
    sentence_mentions_raw: list[list[str]] = []
    mention_to_sentence_set: dict[str, set[int]] = {}

    for sentence_id, sentence in enumerate(sentences):
        mentions = extract_mentions(sentence, min_chars=min_mention_chars)
        sentence_mentions_raw.append(mentions)

        for mention in mentions:
            mention_to_sentence_set.setdefault(mention, set()).add(sentence_id)

    total_sentences = max(len(sentences), 1)
    raw_mentions = len(mention_to_sentence_set)
    filtered_hubs = 0

    mention_id_map: dict[str, int] = {}
    mention_texts: list[str] = []
    mention_to_sentences: dict[int, list[int]] = {}
    mention_df: list[int] = []

    for mention_text, sentence_ids in mention_to_sentence_set.items():
        df = len(sentence_ids)
        if df / total_sentences > max_mention_df_ratio:
            filtered_hubs += 1
            continue

        mention_id = len(mention_texts)
        mention_id_map[mention_text] = mention_id
        mention_texts.append(mention_text)
        mention_to_sentences[mention_id] = sorted(sentence_ids)
        mention_df.append(df)

    sentence_mentions: dict[int, list[int]] = {}
    for sentence_id, mentions in enumerate(sentence_mentions_raw):
        mention_ids = [mention_id_map[m] for m in mentions if m in mention_id_map]
        if mention_ids:
            sentence_mentions[sentence_id] = mention_ids

    graph_index = {
        "sentence_to_chunk": sentence_to_chunk,
        "sentence_mentions": sentence_mentions,
        "mention_texts": mention_texts,
        "mention_to_sentences": mention_to_sentences,
        "mention_df": mention_df,
    }
    stats = {
        "raw_mentions": raw_mentions,
        "kept_mentions": len(mention_texts),
        "filtered_hubs": filtered_hubs,
    }
    return graph_index, stats


def build_index(
    chunks_file: str, output_dir: str, model_name: str, device: str = None, batch_size: int = 32
):
    """Build sentence-level embedding index."""
    from sentence_transformers import SentenceTransformer

    # Load chunks
    print(f"Loading chunks from: {chunks_file}")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks")

    # Create chunk lookup
    chunk_lookup = {c["id"]: c for c in chunks}

    # Extract sentences
    print("Extracting sentences...")
    sentences = []
    sentence_to_chunk = []

    for chunk in track(chunks, description="Extracting sentences"):
        chunk_sentences = split_sentences(chunk["text"])
        for sent in chunk_sentences:
            sentences.append(sent)
            sentence_to_chunk.append(chunk["id"])

    print(f"Total sentences: {len(sentences)}")

    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Encode sentences
    print("Encoding sentences...")
    embeddings = model.encode(
        sentences, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )

    min_mention_chars = get_env_int("ARAG_MIN_MENTION_CHARS", 4)
    max_mention_df_ratio = get_env_float("ARAG_MAX_MENTION_DF_RATIO", 0.02)
    print(
        "Building graph index "
        f"(min_chars={min_mention_chars}, max_df_ratio={max_mention_df_ratio})..."
    )
    graph_index_data, graph_stats = build_graph_index(
        sentences=sentences,
        sentence_to_chunk=sentence_to_chunk,
        min_mention_chars=min_mention_chars,
        max_mention_df_ratio=max_mention_df_ratio,
    )

    # Save index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "sentence_index.pkl"
    graph_index_file = output_path / "graph_index.pkl"
    index_data = {
        "sentences": sentences,
        "embeddings": embeddings,
        "sentence_to_chunk": sentence_to_chunk,
        "chunks": chunk_lookup,
        "model_name": model_name,
    }

    print(f"Saving index to: {index_file}")
    with open(index_file, "wb") as f:
        pickle.dump(index_data, f)

    print(f"Saving graph index to: {graph_index_file}")
    with open(graph_index_file, "wb") as f:
        pickle.dump(graph_index_data, f)

    print("Index built successfully!")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Sentences: {len(sentences)}")
    print(f"  - Embedding dim: {embeddings.shape[1]}")
    print(f"  - Raw mentions: {graph_stats['raw_mentions']}")
    print(f"  - Kept mentions: {graph_stats['kept_mentions']}")
    print(f"  - Filtered hub mentions: {graph_stats['filtered_hubs']}")


def main():
    parser = argparse.ArgumentParser(description="Build semantic search index")
    parser.add_argument("--dataset", "-d", default="demo", help="Dataset name")
    parser.add_argument(
        "--model",
        "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name or path",
    )
    parser.add_argument("--device", default=None, help="Device (cuda:0, cpu, etc.)")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    from arag import Config
    config = Config.from_file("configs/local.toml")
    data = config["data"][args.dataset]

    build_index(
        chunks_file=data["chunks_file"],
        output_dir=data["index_dir"],
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
