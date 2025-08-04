import os
import json
import glob
import hashlib
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as np

# ANN
import faiss

# Sparse retrieval
from rank_bm25 import BM25Okapi

# OpenAI for generation
from openai import OpenAI

# Google Generative AI (Gemini) for embeddings
import google.generativeai as genai

# Sentence splitting for embedding-aware chunking
import nltk
from nltk.tokenize import sent_tokenize

load_dotenv()

# Ensure nltk punkt is available (safe to call repeatedly; it no-ops if already installed)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


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
    # You can use sentence-aware chunking by character budget instead of token count.
    chunk_chars: int = 1800            # approx 350-450 tokens worth of text
    chunk_sent_overlap: int = 1        # sentence overlap between chunks
    use_sentence_chunking: bool = True # turn on embedding-aware chunking

    # Legacy word-chunking fallback (unused if use_sentence_chunking=True)
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval sizes
    top_k_dense: int = 10
    top_k_bm25: int = 10
    top_k_final: int = 6
    alpha_dense: float = 0.6  # dense vs sparse weighting

    # Normalization
    normalize_dense_scores: bool = True
    normalize_sparse_scores: bool = True

    # Prompt sizing
    max_context_chars: int = 12000

    # FAISS index choice
    use_hnsw: bool = True   # if True, use HNSW; else IndexFlatIP

    # Dense retrieval enhancements
    dense_overfetch: int = 4          # over-fetch factor before MMR (e.g., 4x)
    mmr_lambda: float = 0.7           # trade-off relevance vs diversity
    query_expansion_neighbors: int = 8  # initial neighbors to expand query
    query_expansion_weight: float = 0.35

    # Final selection constraints
    per_doc_cap: int = 3  # max chunks from same source in final fused list

    # BM25 tokenization
    use_advanced_bm25_tokenization: bool = True


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

def file_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def sentence_chunk_text(text: str, max_chars: int = 1800, overlap_sentences: int = 1) -> List[str]:
    # Clean excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sents = sent_tokenize(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences > 0 else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def word_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
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

# Advanced BM25 tokenization
STOPWORDS = set([
    "the","a","an","and","or","but","if","then","so","of","on","in","to","for","with","by","at","from","as","is","are","was","were","be","been","being",
    "it","its","this","that","these","those","i","you","he","she","they","we","them","his","her","their","our","your",
])

def tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in STOPWORDS]
    return toks

# Maximal Marginal Relevance (MMR)
def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, doc_ids: List[int], k: int, lambda_mult: float = 0.7) -> List[int]:
    selected = []
    candidates = list(range(len(doc_ids)))
    if not candidates:
        return []
    # rel: cosine similarity (dot product since vectors are normalized)
    rel = doc_vecs @ query_vec
    selected_vecs = []
    for _ in range(min(k, len(candidates))):
        best_idx = None
        best_score = -1e9
        for ci in candidates:
            redundancy = 0.0
            if selected_vecs:
                redundancy = max(float(doc_vecs[ci] @ sv) for sv in selected_vecs)
            score = lambda_mult * float(rel[ci]) - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_idx = ci
        selected.append(doc_ids[best_idx])
        selected_vecs.append(doc_vecs[best_idx])
        candidates.remove(best_idx)
    return selected


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
                emb = getattr(resp, "embedding", None)
                if emb is None and isinstance(resp, dict):
                    emb = resp.get("embedding", None)
                if emb is None:
                    emb = resp
                if isinstance(emb, dict):
                    values = emb.get("values", emb.get("embedding", emb.get("data")))
                elif hasattr(emb, "values"):
                    values = emb.values
                elif isinstance(emb, list):
                    values = emb
                else:
                    values = getattr(emb, "embedding", None)
                    if values is None:
                        raise TypeError(f"Unexpected embedding type: {type(emb)}")
                return np.asarray(values, dtype=np.float32)
            except Exception as e:
                last_err = e
        raise last_err

    def encode(self, texts):
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
        self.hash_path = os.path.join(cfg.artifacts_dir, "hashes.json")

    # -----------------------
    # Build / Load
    # -----------------------
    def build_or_load(self):
        docs = read_all_text_files(self.cfg.data_dir)
        if not docs:
            raise RuntimeError(f"No .txt files found under {self.cfg.data_dir}")

        # Compute current hashes
        cur_hashes = {path: file_content_hash(txt) for path, txt in docs.items()}

        if self._artifacts_exist():
            prev_hashes = self._load_hashes()
            if prev_hashes != cur_hashes:
                # Invalidate and rebuild if any file changed
                self._build_from_scratch(docs)
                self._save_artifacts(cur_hashes)
            else:
                self._load_artifacts()
        else:
            self._build_from_scratch(docs)
            self._save_artifacts(cur_hashes)

        # Build BM25
        with open(self.bm25_tok_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        self.bm25 = BM25Okapi(tokens)

        # FAISS
        self._create_faiss()

    def _build_from_scratch(self, docs: Dict[str, str]):
        corpus_chunks = []
        chunk_meta = []
        for path, text in docs.items():
            if self.cfg.use_sentence_chunking:
                chunks = sentence_chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_sent_overlap)
            else:
                chunks = word_chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap)
            for idx, ch in enumerate(chunks):
                if not ch.strip():
                    continue
                corpus_chunks.append(ch)
                chunk_meta.append({"source_path": path, "chunk_id": idx})
        self.corpus_chunks = corpus_chunks
        self.chunk_meta = chunk_meta

        # Dense embeddings via Gemini
        if self.corpus_chunks:
            self.chunk_embeddings = self.embedder.encode(self.corpus_chunks).astype(np.float32)
            norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True) + 1e-12
            self.chunk_embeddings = self.chunk_embeddings / norms
        else:
            self.chunk_embeddings = np.zeros((0, 768), dtype=np.float32)  # safe default

        # BM25 tokens
        if self.cfg.use_advanced_bm25_tokenization:
            tokenized = [tokenize_for_bm25(c) for c in self.corpus_chunks]
        else:
            tokenized = [c.split() for c in self.corpus_chunks]
        self._tokenized = tokenized

    def _save_artifacts(self, hashes: Dict[str, str]):
        np.save(self.embeddings_path, self.chunk_embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_chunks, f, ensure_ascii=False)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_meta, f, ensure_ascii=False)
        with open(self.bm25_tok_path, "w", encoding="utf-8") as f:
            json.dump(self._tokenized, f, ensure_ascii=False)
        with open(self.hash_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, ensure_ascii=False)

    def _artifacts_exist(self) -> bool:
        return all([
            os.path.exists(self.embeddings_path),
            os.path.exists(self.chunks_path),
            os.path.exists(self.meta_path),
            os.path.exists(self.bm25_tok_path),
            os.path.exists(self.hash_path),
        ])

    def _load_artifacts(self):
        self.chunk_embeddings = np.load(self.embeddings_path)
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.corpus_chunks = json.load(f)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.chunk_meta = json.load(f)

    def _load_hashes(self) -> Dict[str, str]:
        if not os.path.exists(self.hash_path):
            return {}
        with open(self.hash_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_faiss(self):
        if len(self.corpus_chunks) == 0:
            # empty index
            self.faiss_index = None
            return
        dim = self.chunk_embeddings.shape[1]
        if self.cfg.use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            # We use cosine via inner product on unit vectors
            # faiss IndexHNSWFlat uses L2 by default; to use IP, wrap with IndexPreTransform or switch metric
            # Simpler approach: keep L2 but unit-normalize; IP ≈ 1 - 0.5*L2^2 monotonic. For simplicity, we stick with default.
            index.add(self.chunk_embeddings.astype(np.float32))
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(self.chunk_embeddings.astype(np.float32))
        self.faiss_index = index

    # -----------------------
    # Retrieval
    # -----------------------
    def _expand_query_embedding(self, q_emb: np.ndarray) -> np.ndarray:
        # Use a quick search to get neighbors, average them with q_emb
        if self.faiss_index is None or len(self.corpus_chunks) == 0:
            return q_emb
        k = min(self.cfg.query_expansion_neighbors, len(self.corpus_chunks))
        if k <= 0:
            return q_emb
        scores, ids = self.faiss_index.search(q_emb[None, :], k)
        neighbor_ids = [int(i) for i in ids[0] if i != -1]
        if not neighbor_ids:
            return q_emb
        neigh = self.chunk_embeddings[neighbor_ids]
        avg = np.mean(neigh, axis=0)
        new_q = (1 - self.cfg.query_expansion_weight) * q_emb + self.cfg.query_expansion_weight * avg
        new_q /= (np.linalg.norm(new_q, keepdims=True) + 1e-12)
        return new_q

    def _dense_search(self, query: str, top_k: int):
        if len(self.corpus_chunks) == 0:
            return []
        q_emb = self.embedder.encode([query]).astype(np.float32)[0]
        q_emb /= (np.linalg.norm(q_emb, keepdims=True) + 1e-12)

        # Query expansion (pseudo-relevance feedback)
        q_emb = self._expand_query_embedding(q_emb)

        # Over-fetch to allow MMR diversification
        k = min(top_k * self.cfg.dense_overfetch, len(self.corpus_chunks))
        scores, ids = self.faiss_index.search(q_emb[None, :], k)
        ids_list = [int(x) for x in ids[0] if x != -1]
        if not ids_list:
            return []

        # MMR: diversify among the over-fetched candidates
        doc_vecs = self.chunk_embeddings[ids_list]  # already normalized
        selected_ids = mmr(q_emb, doc_vecs, ids_list, k=top_k, lambda_mult=self.cfg.mmr_lambda)

        # Map back to original FAISS scores (optional)
        id_to_score = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i != -1}
        return [(i, id_to_score.get(i, 0.0)) for i in selected_ids]

    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.cfg.use_advanced_bm25_tokenization:
            toks = tokenize_for_bm25(query)
        else:
            toks = query.split()
        if len(self.corpus_chunks) == 0:
            return []
        scores = self.bm25.get_scores(toks)
        N = len(scores)
        if N == 0:
            return []
        k = min(top_k, N)
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

        # Per-document cap to increase diversity
        capped = []
        per_doc_count = {}
        for doc_id, score in ranked:
            src = self.chunk_meta[doc_id]["source_path"]
            c = per_doc_count.get(src, 0)
            if c >= self.cfg.per_doc_cap:
                continue
            per_doc_count[src] = c + 1
            capped.append((doc_id, score))
            if len(capped) >= self.cfg.top_k_final:
                break

        return capped

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
            "If the answer cannot be found in the context, say you don't know. Cite sources like [1], [2]. "
            "Avoid unsupported claims."
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

        # Embedding-aware chunking
        use_sentence_chunking=True,
        chunk_chars=1800,
        chunk_sent_overlap=1,

        # Retrieval
        top_k_dense=15,
        top_k_bm25=20,
        top_k_final=8,
        alpha_dense=0.6,

        # Enhancements
        dense_overfetch=4,
        mmr_lambda=0.7,
        query_expansion_neighbors=8,
        query_expansion_weight=0.35,
        per_doc_cap=3,
        use_advanced_bm25_tokenization=True,

        # FAISS
        use_hnsw=False,  # set True for large corpora

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