"""Position-based retrieval and context reconstruction.

Uses TF-IDF to find the most relevant position in the anchor sequence,
then slices a temporal window around it. The hit anchor receives a PRIMARY
marker to distinguish causal from temporal adjacency.

Hybrid retrieval: TF-IDF position search + SQLite FTS5 fallback for
low-confidence queries (handles Chinese synonym zero-recall problem).

Scheme 4 (Constraint Self-Correction): entity overlap + temporal proximity
create cross-validation — the LLM sees multiple anchors about the same entity
and can distinguish real causality from coincidental adjacency.
"""

import math
import re
from collections import Counter
from typing import Optional

from .models import Anchor, AnchorSequence
from .formatter import format_compact

# Minimum TF-IDF score before falling back to SQLite FTS5
HYBRID_FALLBACK_THRESHOLD = 0.15

# Keywords worth keeping in FTS5 queries (filtered from common query words)
_QUERY_STOP_WORDS = {
    "what", "is", "the", "a", "an", "are", "was", "were", "did", "do", "does",
    "how", "when", "where", "who", "why", "which", "and", "or", "in", "of",
    "to", "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "use", "used", "using", "happened", "found", "discovered",
}


def _extract_fts_query(user_query: str) -> str:
    """Extract meaningful keywords from a natural language query for FTS5.

    Removes common stop words and keeps technical terms, proper nouns,
    and domain-specific words that are likely to match anchor entities.
    """
    words = re.findall(r'[A-Za-z0-9_.+-]+', user_query)
    keywords = [w for w in words
                if w.lower() not in _QUERY_STOP_WORDS
                and len(w) > 1]
    # FTS5 MATCH syntax: words OR'd together
    if not keywords:
        return ""
    return " OR ".join(keywords[:6])  # Max 6 keywords to avoid query explosion


class SequenceRetriever:
    """Position-based anchor retriever using TF-IDF similarity.

    Not semantic search — just TF-IDF to find WHERE in the sequence
    a query is most relevant. Returns a positional window, not ranked results.
    """

    def __init__(self, sequence: AnchorSequence):
        self.sequence = sequence
        self.active = sequence.get_active()
        # Build search text: entity + tags (for semantic matching)
        # Tags ARE used for search but NOT stored in entity text
        self._anchor_texts = []
        for a in self.active:
            text = a.entity
            if a.tags:
                text += " " + " ".join(a.tags)
            self._anchor_texts.append(text)
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        """Compute IDF scores for terms across all anchors."""
        n = len(self._anchor_texts)
        if n == 0:
            return {}

        df: dict[str, int] = {}
        for text in self._anchor_texts:
            terms = set(self._tokenize(text))
            for term in terms:
                df[term] = df.get(term, 0) + 1

        return {term: math.log((n + 1) / (freq + 1)) + 1.0
                for term, freq in df.items()}

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer — splits on spaces, punctuation, case boundaries."""
        tokens = re.findall(r'[一-鿿]+|[a-zA-Z]+|\d+', text.lower())
        return tokens

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for a text string."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1

        vec = {}
        for term, count in tf.items():
            tf_norm = count / max_tf
            idf = self._idf.get(term, 1.0)
            vec[term] = tf_norm * idf

        return vec

    def _cosine_similarity(self, v1: dict[str, float], v2: dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        if not v1 or not v2:
            return 0.0

        dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in set(v1) | set(v2))
        mag1 = math.sqrt(sum(v * v for v in v1.values()))
        mag2 = math.sqrt(sum(v * v for v in v2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def find_position(self, query: str) -> tuple[int, int, float]:
        """Find the most relevant anchor position for a query.

        Uses TF-IDF cosine similarity. When TF-IDF returns zero (no token
        overlap), falls back to simple substring matching against anchor
        entities to avoid returning meaningless position 0.

        Returns:
            (sequence_index, anchor_index, score)
        """
        if not self.active:
            raise ValueError("No active anchors available for retrieval")

        query_vec = self._tfidf_vector(query)
        best_idx = 0
        best_score = 0.0

        if query_vec:
            for i, text in enumerate(self._anchor_texts):
                anchor_vec = self._tfidf_vector(text)
                score = self._cosine_similarity(query_vec, anchor_vec)
                if score > best_score:
                    best_score = score
                    best_idx = i

        # Fallback: when TF-IDF gives zero, try keyword substring matching
        if best_score == 0.0:
            keywords = _extract_fts_query(query).split(" OR ")
            for kw in keywords:
                kw_lower = kw.lower()
                for i, text in enumerate(self._anchor_texts):
                    if kw_lower in text.lower():
                        best_idx = i
                        best_score = 0.1  # Low but non-zero: "keyword match"
                        break
                if best_score > 0:
                    break

        return (0, best_idx, best_score)

    def get_window(self, center_index: int, radius: int = 2) -> list[Anchor]:
        """Get a positional window around center_index."""
        return self.sequence.get_window(center_index, radius)

    def build_reconstruction_prompt(
        self,
        query: str,
        radius: int = 2,
        include_constraints: bool = True,
        fts_results: Optional[list[dict]] = None,
    ) -> str:
        """Build a reconstruction prompt for the LLM.

        The prompt includes:
        1. A positional window of anchors around the query hit
        2. A PRIMARY marker on the hit anchor to distinguish it
        3. Optional constraint graph for cross-validation
        4. Optional FTS5 fallback results for low-confidence queries

        Args:
            query: The user's question or topic.
            radius: Number of neighbors on each side of the hit.
            include_constraints: Whether to include constraint edges.
            fts_results: Optional FTS5 search results for hybrid retrieval.

        Returns:
            A prompt string for the LLM to reconstruct context from anchors.
        """
        _, hit_idx, score = self.find_position(query)
        window = self.get_window(hit_idx, radius)

        # Re-map hit_idx to the window's coordinate system
        window_hit_idx = -1
        for i, a in enumerate(window):
            if a.pos == self.active[hit_idx].pos:
                window_hit_idx = i
                break

        retrieval_method = "TF-IDF positional"
        if score < HYBRID_FALLBACK_THRESHOLD and fts_results:
            retrieval_method += " + FTS5 semantic fallback"

        lines = [
            "# Context Reconstruction from Anchors",
            "",
            f"**Query:** {query}",
            f"**Match confidence:** {score:.2f}  ({retrieval_method})",
            "",
            "## Anchor Window (Position-Based)",
            "",
            "The anchor marked ★ PRIMARY is the direct match for your query.",
            "Adjacent anchors are temporally nearby — they may or may not be causally related.",
            "Distinguish between:",
            "- **Causal relationship:** The PRIMARY anchor was caused by or caused a neighbor",
            "- **Temporal adjacency:** They just happened at similar times in the conversation",
            "",
            format_compact(self.sequence, window_hit_idx),
        ]

        # Add FTS5 fallback results if TF-IDF was low confidence
        if score < HYBRID_FALLBACK_THRESHOLD and fts_results:
            lines.append("")
            lines.append("## Semantic Matches (FTS5 Fallback)")
            lines.append("")
            lines.append("The position-based match had low confidence. These semantically")
            lines.append("similar anchors from other sessions may be relevant:")
            lines.append("")
            for r in fts_results[:5]:
                data_str = ""
                if r.get("data_values"):
                    dv = r["data_values"]
                    if isinstance(dv, list) and dv:
                        data_str = f" [{', '.join(dv)}]"
                lines.append(f"  • [{r['anchor_type']}] {r['entity']}{data_str}  (session: {r['session_id'][:8]})")

        if include_constraints:
            from .constraints import build_constraint_graph
            graph = build_constraint_graph(self.sequence)
            if graph["edges"]:
                lines.append("")
                lines.append("## Constraint Relationships")
                for edge in graph["edges"]:
                    from_node = edge["from"]
                    to_node = edge["to"]
                    if from_node < len(self.active) and to_node < len(self.active):
                        lines.append(
                            f"  [{from_node}] {self.active[from_node].entity} "
                            f"--{edge['relation']}--> "
                            f"[{to_node}] {self.active[to_node].entity}"
                        )

        lines.append("")
        lines.append("## Instructions")
        lines.append("Based on the anchor window above, reconstruct the relevant conversation context.")
        lines.append("Focus on the PRIMARY anchor and its causally related neighbors.")
        lines.append("Be explicit about what you can confirm vs. what is uncertain.")

        return "\n".join(lines)


class HybridRetriever:
    """Combines TF-IDF position retrieval with SQLite FTS5 semantic fallback.

    Strategy:
      1. Try TF-IDF first (position-based, preserves temporal context)
      2. If score < threshold, search SQLite FTS5 for semantic matches
      3. Return both position window + semantic results
    """

    def __init__(self, sequence: AnchorSequence):
        self.tfidf = SequenceRetriever(sequence)
        self.sequence = sequence
        self._sqlite = None

    @property
    def sqlite(self):
        if self._sqlite is None:
            try:
                from .store_sqlite import SqliteStore
                self._sqlite = SqliteStore()
            except Exception:
                self._sqlite = False  # Sentinel: tried and failed
        return self._sqlite if self._sqlite is not False else None

    def search(self, query: str, radius: int = 2) -> str:
        """Hybrid search: TF-IDF position + SQLite FTS5 fallback.

        When TF-IDF confidence is low, preprocesses the query into
        keywords before FTS5 search (instead of sending raw natural
        language which confuses FTS5 MATCH syntax).

        Returns a reconstruction prompt string.
        """
        seq_idx, hit_idx, score = self.tfidf.find_position(query)

        fts_results = None
        if score < HYBRID_FALLBACK_THRESHOLD and self.sqlite:
            try:
                fts_query = _extract_fts_query(query)
                if fts_query:
                    fts_results = self.sqlite.search(fts_query, limit=5)
            except Exception:
                pass

        return self.tfidf.build_reconstruction_prompt(
            query, radius=radius, fts_results=fts_results
        )
