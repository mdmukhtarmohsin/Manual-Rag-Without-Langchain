import os
import json
import glob
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# ANN
import faiss

# Sparse retrieval
from rank_bm25 import BM25Okapi

# OpenAI for generation
from openai import OpenAI

# Google Generative AI (Gemini) for embeddings
import google.generativeai as genai

load_dotenv()


# ---------------------------
# Config
# ---------------------------
@dataclass
class RAGConfig:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    # Embeddings: Gemini
    gemini_embedding_model: str = "models/text-embedding-004"

    # LLM: OpenAI
    openai_model_name: str = "gpt-4o-mini"

    # Retrieval / chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k_dense: int = 10
    top_k_bm25: int = 10
    top_k_final: int = 6
    alpha_dense: float = 0.6  # dense vs sparse weighting

    # Normalization
    normalize_dense_scores: bool = True
    normalize_sparse_scores: bool = True

    # Prompt sizing
    max_context_chars: int = 12000


# ---------------------------
# Utils
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_all_text_files(data_dir: str) -> Dict[str, str]:
    docs = {}
    for path in glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs[path] = f.read()
    return docs

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += step
    return chunks

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smax - smin < 1e-12:
        return np.ones_like(scores)
    return (scores - smin) / (smax - smin)

def reciprocal_rank_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
    fused = {}
    for ranking in rankings:
        for rank_idx, (doc_id, _) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return fused


# ---------------------------
# Embedding client (Gemini)
# ---------------------------
class GeminiEmbedder:
    def __init__(self, model: str = "models/text-embedding-004", api_key: str = None, batch_size: int = 64, retry: int = 3):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.retry = retry

    def _embed_one(self, text: str) -> np.ndarray:
        last_err = None
        for _ in range(self.retry):
            try:
                resp = genai.embed_content(model=self.model, content=text)

                # Common shapes seen across SDK versions:
                # 1) resp = {"embedding": {"values": [...]}}
                # 2) resp.embedding.values -> list[float]
                # 3) resp["embedding"] -> list[float]
                # 4) resp.embedding -> list[float]
                # 5) resp -> list[float] (rare)

                # Get "embedding" container first (object or dict or list)
                emb = getattr(resp, "embedding", None)
                if emb is None and isinstance(resp, dict):
                    emb = resp.get("embedding", None)
                if emb is None:
                    emb = resp  # sometimes the response is just the vector

                # Extract list[float] from the container
                if isinstance(emb, dict):
                    values = emb.get("values", emb.get("embedding", emb.get("data")))
                elif hasattr(emb, "values"):  # object with .values
                    values = emb.values
                elif isinstance(emb, list):   # already a list of floats
                    values = emb
                else:
                    # last resort: try to index into known fields
                    values = getattr(emb, "embedding", None)
                    if values is None:
                        raise TypeError(f"Unexpected embedding type: {type(emb)}")

                return np.asarray(values, dtype=np.float32)
            except Exception as e:
                last_err = e
        raise last_err

    def encode(self, texts):
        # Ensure list[str]
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            for t in batch:
                vecs.append(self._embed_one(t))
        return np.vstack(vecs)


# ---------------------------
# RAG Pipeline
# ---------------------------
class ManualRAG:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        ensure_dir(cfg.artifacts_dir)

        # Initialize embedding + generation clients
        self.embedder = GeminiEmbedder(model=cfg.gemini_embedding_model)
        self.openai = OpenAI()  # uses OPENAI_API_KEY

        # Index holders
        self.faiss_index = None
        self.bm25 = None
        self.corpus_chunks = []
        self.chunk_meta = []
        self.chunk_embeddings = None

        # Artifacts
        self.embeddings_path = os.path.join(cfg.artifacts_dir, "embeddings.npy")
        self.chunks_path = os.path.join(cfg.artifacts_dir, "chunks.json")
        self.meta_path = os.path.join(cfg.artifacts_dir, "meta.json")
        self.bm25_tok_path = os.path.join(cfg.artifacts_dir, "bm25_tokens.json")

    # -----------------------
    # Build / Load
    # -----------------------
    def build_or_load(self):
        if self._artifacts_exist():
            self._load_artifacts()
        else:
            self._build_from_scratch()
            self._save_artifacts()

        with open(self.bm25_tok_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        self.bm25 = BM25Okapi(tokens)

        self._create_faiss()

    def _build_from_scratch(self):
        docs = read_all_text_files(self.cfg.data_dir)
        if not docs:
            raise RuntimeError(f"No .txt files found under {self.cfg.data_dir}")

        corpus_chunks = []
        chunk_meta = []
        for path, text in docs.items():
            chunks = chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for idx, ch in enumerate(chunks):
                corpus_chunks.append(ch)
                chunk_meta.append({"source_path": path, "chunk_id": idx})
        self.corpus_chunks = corpus_chunks
        self.chunk_meta = chunk_meta

        # Dense embeddings via Gemini
        self.chunk_embeddings = self.embedder.encode(self.corpus_chunks).astype(np.float32)
        # Normalize embeddings to unit length for cosine via inner product
        norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True) + 1e-12
        self.chunk_embeddings = self.chunk_embeddings / norms

        # BM25 tokens
        tokenized = [c.split() for c in self.corpus_chunks]
        self._tokenized = tokenized

    def _save_artifacts(self):
        np.save(self.embeddings_path, self.chunk_embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_chunks, f, ensure_ascii=False)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_meta, f, ensure_ascii=False)
        with open(self.bm25_tok_path, "w", encoding="utf-8") as f:
            json.dump(self._tokenized, f, ensure_ascii=False)

    def _artifacts_exist(self) -> bool:
        return all([
            os.path.exists(self.embeddings_path),
            os.path.exists(self.chunks_path),
            os.path.exists(self.meta_path),
            os.path.exists(self.bm25_tok_path),
        ])

    def _load_artifacts(self):
        self.chunk_embeddings = np.load(self.embeddings_path)
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.corpus_chunks = json.load(f)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.chunk_meta = json.load(f)

    def _create_faiss(self):
        dim = self.chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self.chunk_embeddings.astype(np.float32))
        self.faiss_index = index

    # -----------------------
    # Retrieval
    # -----------------------
    def _dense_search(self, query: str, top_k: int):
        q_emb = self.embedder.encode([query]).astype(np.float32)
        q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        # Ask FAISS for up to available docs
        k = min(top_k, len(self.corpus_chunks))
        scores, ids = self.faiss_index.search(q_emb, k)
        valid = [(int(ids[0][i]), float(scores[0][i])) for i in range(len(ids[0])) if ids[0][i] != -1]
        return valid

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        toks = query.split()
        scores = self.bm25.get_scores(toks)
        N = len(scores)
        if N == 0:
            return []
        k = min(top_k, N)
        # If k == N, just sort all; else use argpartition for efficiency
        if k == N:
            idx = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        return [(int(i), float(scores[i])) for i in idx]

    def hybrid_search(self, query: str) -> List[Tuple[int, float]]:
        dense = self._dense_search(query, self.cfg.top_k_dense)
        sparse = self._sparse_search(query, self.cfg.top_k_bm25)

        if self.cfg.normalize_dense_scores and dense:
            d = np.array([s for _, s in dense])
            d = min_max_normalize(d)
            dense = [(i, float(s)) for (i, _), s in zip(dense, d)]
        if self.cfg.normalize_sparse_scores and sparse:
            s = np.array([s for _, s in sparse])
            s = min_max_normalize(s)
            sparse = [(i, float(v)) for (i, _), v in zip(sparse, s)]

        dense_sorted = sorted(dense, key=lambda x: x[1], reverse=True)
        sparse_sorted = sorted(sparse, key=lambda x: x[1], reverse=True)

        rrf = reciprocal_rank_fusion([dense_sorted, sparse_sorted], k=60)

        weighted = {}
        for doc_id, sc in dense_sorted:
            weighted[doc_id] = weighted.get(doc_id, 0.0) + self.cfg.alpha_dense * sc
        for doc_id, sc in sparse_sorted:
            weighted[doc_id] = weighted.get(doc_id, 0.0) + (1 - self.cfg.alpha_dense) * sc

        fused = {}
        for doc_id in set(list(weighted.keys()) + list(rrf.keys())):
            fused[doc_id] = weighted.get(doc_id, 0.0) + 0.25 * rrf.get(doc_id, 0.0)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[: self.cfg.top_k_final]

    # -----------------------
    # Generation (OpenAI)
    # -----------------------
    def _truncate_contexts(self, contexts: List[str], max_chars: int) -> List[str]:
        out, total = [], 0
        for c in contexts:
            if total + len(c) + 50 > max_chars:
                remaining = max(0, max_chars - total - 10)
                if remaining > 200:
                    out.append(c[:remaining])
                break
            out.append(c)
            total += len(c)
        return out

    def generate_answer(self, query: str, contexts: List[str], temperature: float = 0.2, max_tokens: int = 512) -> str:
        contexts = self._truncate_contexts(contexts, self.cfg.max_context_chars)
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        system = (
            "You are a careful and concise assistant. Use ONLY the provided context to answer. "
            "If the answer cannot be found, say you don't know. Cite sources like [1], [2]."
        )
        user = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            f"Answer with citations:"
        )
        resp = self.openai.chat.completions.create(
            model=self.cfg.openai_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    # -----------------------
    # Public API
    # -----------------------
    def query(self, question: str) -> Dict:
        ranked = self.hybrid_search(question)
        retrieved = []
        for doc_id, score in ranked:
            retrieved.append({
                "doc_id": doc_id,
                "score": score,
                "text": self.corpus_chunks[doc_id],
                "meta": self.chunk_meta[doc_id],
            })
        contexts = [r["text"] for r in retrieved]
        answer = self.generate_answer(question, contexts)
        return {"question": question, "answer": answer, "contexts": retrieved}


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    cfg = RAGConfig(
        data_dir="data",
        artifacts_dir="artifacts",
        gemini_embedding_model="models/text-embedding-004",
        openai_model_name="gpt-4o-mini",
        chunk_size=500,
        chunk_overlap=100,
        top_k_dense=15,
        top_k_bm25=20,
        top_k_final=8,
        alpha_dense=0.6,
        max_context_chars=12000,
    )

    rag = ManualRAG(cfg)
    rag.build_or_load()

    print("Enter a question (or 'exit'):")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = rag.query(q)
        print("\nAnswer:\n", result["answer"])
        print("\nCitations:")
        for i, r in enumerate(result["contexts"], 1):
            print(f"[{i}] {r['meta']['source_path']} (chunk {r['meta']['chunk_id']}) | score={r['score']:.4f}")
        print()