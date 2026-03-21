"""
02_build_rag_index.py
---------------------
Builds a FAISS vector index from:
  1. Fine-tuned PerturbationCatalogueLLM perturbation embeddings  (from step 01)
  2. Rich text summaries for each perturbation (for LLM context injection)

The index stores both:
  - Dense vectors  → nearest-neighbour retrieval
  - Text documents → injected as context into the LLM prompt

USAGE:
    python 02_build_rag_index.py \
        --data_path       data/perturb_seq.csv \
        --embeddings_dir  embeddings/perturbation_catalogue_llm \
        --output_dir      embeddings/rag_index
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("⚠  faiss not installed. Falling back to numpy cosine similarity.")
    print("   Install with: pip install faiss-cpu")


# ════════════════════════════════════════════════════════════════════════════
# 1. BUILD TEXT DOCUMENTS FROM PERTURB-SEQ DATA
# ════════════════════════════════════════════════════════════════════════════

def build_documents(csv_path: str, top_n: int = 20) -> list[dict]:
    """
    Convert each perturbation's differential expression results into a
    rich natural-language document for RAG context injection.

    Returns list of dicts:
        {
          "perturbation": "TP53",
          "text": "Knocking out TP53 causes significant up-regulation of ...",
          "metadata": { top_up: [...], top_down: [...], n_sig: int }
        }
    """
    print(f"Building RAG documents from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["Log2FC"]    = pd.to_numeric(df["Log2FC"],    errors="coerce").fillna(0.0)
    df["Padj"]      = pd.to_numeric(df["Padj"],      errors="coerce").fillna(1.0)
    df["Score Value"] = pd.to_numeric(df["Score Value"], errors="coerce").fillna(0.0)

    documents = []

    for pert_gene, group in df.groupby("Perturbation Gene"):
        sig = group[group["Padj"] < 0.05].copy()
        sig = sig.sort_values("Score Value", key=abs, ascending=False)

        up   = sig[sig["Log2FC"] > 0].head(top_n)
        down = sig[sig["Log2FC"] < 0].head(top_n)

        def format_genes(subset):
            parts = []
            for _, row in subset.iterrows():
                direction = "↑" if row["Log2FC"] > 0 else "↓"
                parts.append(
                    f"{row['Effect Gene']} (Log2FC={row['Log2FC']:.2f}, "
                    f"Padj={row['Padj']:.2e}, score={row['Score Value']:.2f})"
                )
            return parts

        up_genes   = format_genes(up)
        down_genes = format_genes(down)

        # Build the natural-language context document
        cell_type = sig["Cell Type"].dropna().unique()
        cell_str  = f" in {', '.join(cell_type)}" if len(cell_type) > 0 else ""

        text = f"""Perturbation: knockout of {pert_gene}{cell_str}

Summary:
- Total significant differentially expressed genes (Padj < 0.05): {len(sig):,}
- Up-regulated genes: {len(up[up['Log2FC'] > 0]):,}
- Down-regulated genes: {len(down[down['Log2FC'] < 0]):,}

Top up-regulated genes (strongest positive Log2FC):
{chr(10).join('  ' + g for g in up_genes) if up_genes else '  None'}

Top down-regulated genes (strongest negative Log2FC):
{chr(10).join('  ' + g for g in down_genes) if down_genes else '  None'}

Interpretation:
Knocking out {pert_gene} leads to {'up-regulation of ' + ', '.join(up['Effect Gene'].head(5).tolist()) if not up.empty else 'no significant up-regulation'}
and {'down-regulation of ' + ', '.join(down['Effect Gene'].head(5).tolist()) if not down.empty else 'no significant down-regulation'}.
The strongest effect is on {sig.iloc[0]['Effect Gene'] if len(sig) > 0 else 'no gene'} 
(Log2FC={sig.iloc[0]['Log2FC']:.2f} if len(sig) > 0 else 'N/A').
"""

        documents.append({
            "perturbation": pert_gene,
            "text":         text,
            "metadata": {
                "n_significant":   len(sig),
                "n_upregulated":   len(sig[sig["Log2FC"] > 0]),
                "n_downregulated": len(sig[sig["Log2FC"] < 0]),
                "top_up_genes":    up["Effect Gene"].head(10).tolist(),
                "top_down_genes":  down["Effect Gene"].head(10).tolist(),
                "max_log2fc":      float(sig["Log2FC"].max()) if len(sig) > 0 else 0.0,
                "min_log2fc":      float(sig["Log2FC"].min()) if len(sig) > 0 else 0.0,
            }
        })

        print(f"  {pert_gene}: {len(sig):,} significant genes "
              f"({len(up)} up, {len(down)} down)")

    return documents


# ════════════════════════════════════════════════════════════════════════════
# 2. BUILD FAISS INDEX
# ════════════════════════════════════════════════════════════════════════════

class PerturbationRAGIndex:
    """
    Stores perturbation embeddings in a FAISS index and associates each
    vector with its text document for context retrieval.

    Usage:
        index = PerturbationRAGIndex.load("embeddings/rag_index")
        results = index.query("What happens when SPI1 is knocked out?", k=3)
    """

    def __init__(self):
        self.embeddings  = None   # np.ndarray (N, D)
        self.documents   = []     # list of dicts
        self.labels      = []     # list of perturbation gene names
        self.faiss_index = None
        self.embed_dim   = None

    def build(self, embeddings: np.ndarray, documents: list[dict]):
        """Build the index from embeddings + documents."""
        assert len(embeddings) == len(documents), \
            f"Embeddings ({len(embeddings)}) and documents ({len(documents)}) must match"

        self.embeddings = embeddings.astype(np.float32)
        self.documents  = documents
        self.labels     = [d["perturbation"] for d in documents]
        self.embed_dim  = embeddings.shape[1]

        # Normalise for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normed = self.embeddings / norms

        if HAS_FAISS:
            # Inner product on normalised vectors = cosine similarity
            self.faiss_index = faiss.IndexFlatIP(self.embed_dim)
            self.faiss_index.add(normed)
            print(f"  FAISS index built: {self.faiss_index.ntotal} vectors, dim={self.embed_dim}")
        else:
            # Fallback: store normalised vectors for numpy cosine search
            self._normed = normed
            print(f"  Numpy index built: {len(normed)} vectors, dim={self.embed_dim}")

    def query_by_vector(self, query_vector: np.ndarray, k: int = 3) -> list[dict]:
        """
        Retrieve top-k documents by embedding similarity.
        query_vector: (D,) numpy array
        """
        q = query_vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm

        if HAS_FAISS and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q, k)
            scores  = scores[0]
            indices = indices[0]
        else:
            sims    = (self._normed @ q.T).squeeze()
            indices = np.argsort(sims)[::-1][:k]
            scores  = sims[indices]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            results.append({
                "perturbation": self.labels[idx],
                "score":        float(score),
                "document":     self.documents[idx],
            })
        return results

    def query_by_name(self, perturbation_name: str) -> dict | None:
        """Direct lookup by perturbation gene name."""
        name = perturbation_name.upper()
        for doc in self.documents:
            if doc["perturbation"].upper() == name:
                return doc
        return None

    def get_all_perturbations(self) -> list[str]:
        return self.labels

    def save(self, output_dir: str):
        """Save index to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "embeddings.npy", self.embeddings)

        with open(output_dir / "documents.json", "w") as f:
            json.dump(self.documents, f, indent=2)

        with open(output_dir / "labels.json", "w") as f:
            json.dump(self.labels, f)

        index_meta = {"embed_dim": self.embed_dim, "n_documents": len(self.documents)}
        with open(output_dir / "index_meta.json", "w") as f:
            json.dump(index_meta, f, indent=2)

        if HAS_FAISS and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(output_dir / "faiss.index"))

        print(f"✅ RAG index saved to {output_dir}")

    @classmethod
    def load(cls, index_dir: str) -> "PerturbationRAGIndex":
        """Load index from disk."""
        index_dir = Path(index_dir)
        obj = cls()

        obj.embeddings = np.load(index_dir / "embeddings.npy")
        obj.embed_dim  = obj.embeddings.shape[1]

        with open(index_dir / "documents.json") as f:
            obj.documents = json.load(f)

        with open(index_dir / "labels.json") as f:
            obj.labels = json.load(f)

        norms = np.linalg.norm(obj.embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normed = (obj.embeddings / norms).astype(np.float32)

        if HAS_FAISS:
            faiss_path = index_dir / "faiss.index"
            if faiss_path.exists():
                obj.faiss_index = faiss.read_index(str(faiss_path))
            else:
                obj.faiss_index = faiss.IndexFlatIP(obj.embed_dim)
                obj.faiss_index.add(normed)
        else:
            obj._normed = normed

        print(f"✅ RAG index loaded: {len(obj.labels)} perturbations from {index_dir}")
        return obj


# ════════════════════════════════════════════════════════════════════════════
# 3. ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    embeddings_dir = Path(args.embeddings_dir)

    # Load embeddings from step 01
    embedding_index_path = embeddings_dir / "embedding_index.json"
    if not embedding_index_path.exists():
        raise FileNotFoundError(
            f"No embedding_index.json found at {embeddings_dir}. "
            "Run 01_finetune.py first to generate PerturbationCatalogueLLM embeddings."
        )

    with open(embedding_index_path) as f:
        embedding_index = json.load(f)

    pert_names = list(embedding_index.keys())
    embeddings = np.array([embedding_index[p] for p in pert_names], dtype=np.float32)
    print(f"Loaded {len(pert_names)} embeddings of dim {embeddings.shape[1]}")

    # Build text documents
    documents = build_documents(args.data_path, top_n=args.top_n)

    # Align documents to embedding order
    doc_map = {d["perturbation"]: d for d in documents}
    aligned_docs = []
    aligned_embs = []
    for name, emb in zip(pert_names, embeddings):
        if name in doc_map:
            aligned_docs.append(doc_map[name])
            aligned_embs.append(emb)
        else:
            print(f"  ⚠ No document found for perturbation: {name}")

    aligned_embs = np.array(aligned_embs, dtype=np.float32)

    # Build and save the index
    rag_index = PerturbationRAGIndex()
    rag_index.build(aligned_embs, aligned_docs)
    rag_index.save(args.output_dir)

    # Quick sanity check
    print("\nSanity check — query for first perturbation:")
    results = rag_index.query_by_vector(aligned_embs[0], k=2)
    for r in results:
        print(f"  {r['perturbation']}  (score={r['score']:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",      default="data/perturb_seq.csv")
    p.add_argument("--embeddings_dir", default="embeddings/perturbation_catalogue_llm")
    p.add_argument("--output_dir",     default="embeddings/rag_index")
    p.add_argument("--top_n",          type=int, default=20,
                   help="Top N genes to include in each document")
    main(p.parse_args())
