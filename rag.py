import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc: str
    text: str


def load_docs(data_dir: str = "data") -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for fn in ["catalogue.md", "faq.md", "policies.md"]:
        path = os.path.join(data_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            docs.append((fn, f.read()))
    return docs


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 60) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size].strip())
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c]


class SimpleRAG:
    """Retrieval simple (TF-IDF + cosine similarity)."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.chunks: List[Chunk] = []
        self.matrix = None

    def build(self, docs: List[Tuple[str, str]]):
        self.chunks = []
        texts: List[str] = []

        for doc_name, content in docs:
            for c in chunk_text(content):
                self.chunks.append(Chunk(doc=doc_name, text=c))
                texts.append(c)

        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.matrix is None:
            raise RuntimeError("Index non construit. Appelez build(load_docs()) d'abord.")

        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]

        out: List[Dict[str, Any]] = []
        for i in idxs:
            out.append(
                {
                    "doc": self.chunks[int(i)].doc,
                    "score": float(sims[int(i)]),
                    "text": self.chunks[int(i)].text,
                }
            )
        return out