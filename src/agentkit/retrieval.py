from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str


def chunk_markdown(doc_id: str, markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    doc_title = doc_id
    for block in re.split(r"(?m)^(?=## )", markdown):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            doc_title = block.splitlines()[0][2:].strip()
            continue
        heading, _, body = block.partition("\n")
        heading = heading.lstrip("# ").strip()
        chunks.append(Chunk(doc_id, f"{doc_id}#{_slug(heading)}", f"{doc_title} / {heading}", body.strip()))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        ranked = sorted(zip(self.score(query), self.chunks, strict=True), key=lambda p: -p[0])
        return [c for s, c in ranked[:k] if s > 0]


def load_index(docs_dir: Path) -> BM25Index:
    chunks = [c for p in sorted(docs_dir.glob("*.md")) for c in chunk_markdown(p.stem, p.read_text())]
    return BM25Index(chunks)
