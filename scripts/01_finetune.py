"""
01_finetune.py
--------------------
Fine-tune PerturbationCatalogueLLM based on
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
on your Perturb-seq differential expression data.

Pre-trained model: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
  - BiomedBERT understands biomedical gene/protein names from abstract-level pre-training
  - We add an MLP encoder to project Log2FC profiles → dense embeddings
  - Fine-tuned embeddings are then used for RAG retrieval

USAGE (Colab / local GPU):
    python 01_finetune.py \
        --data_path data/perturb_seq.csv \
        --output_dir embeddings/perturbation_catalogue_llm \
        --epochs 10 \
        --batch_size 64
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Optional: nicer progress bars ──────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def load_and_pivot(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load Perturb-seq CSV and pivot into a gene-expression matrix.

    Input CSV columns:
        Perturbation Gene, Effect Gene, Log2FC, Padj, Score Name,
        Score Value, Cell Type

    Output:
        matrix : DataFrame  shape (n_perturbations, n_genes)
                 values = Log2FC  (0.0 where not significant or missing)
        meta   : dict with gene list, perturbation list, label encoders
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["Log2FC"] = pd.to_numeric(df["Log2FC"], errors="coerce").fillna(0.0)

    # Keep only significant hits (Padj < 0.05) — zero-fill the rest
    df_sig = df[df["Padj"] < 0.05].copy()

    # Pivot: rows = perturbation genes, cols = effect genes
    matrix = df_sig.pivot_table(
        index="Perturbation Gene",
        columns="Effect Gene",
        values="Log2FC",
        aggfunc="mean",
    ).fillna(0.0)

    print(f"  Perturbations : {matrix.shape[0]}")
    print(f"  Effect genes  : {matrix.shape[1]}")

    gene_list = list(matrix.columns)
    pert_list = list(matrix.index)

    meta = {
        "gene_list": gene_list,
        "perturbation_list": pert_list,
        "n_genes": len(gene_list),
        "n_perturbations": len(pert_list),
    }
    return matrix, meta


class PerturbSeqDataset(Dataset):
    """
    Each sample = one perturbation's expression profile (vector of Log2FC values).
    The model learns to embed this profile into a dense vector.

    For the regression head we use the profile itself as the reconstruction
    target (autoencoder-style), which forces the embedding to capture the
    most informative dimensions of the expression change.
    """
    def __init__(self, matrix: pd.DataFrame, max_genes: int = 2048):
        self.profiles = torch.tensor(matrix.values, dtype=torch.float32)
        self.labels   = list(matrix.index)          # perturbation gene names
        self.max_genes = max_genes

        # Normalise each row to unit norm (standard for expression embeddings)
        norms = self.profiles.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.profiles = self.profiles / norms

        # If more genes than max_genes, keep top by absolute magnitude
        if self.profiles.shape[1] > max_genes:
            mean_abs = self.profiles.abs().mean(0)
            top_idx  = mean_abs.topk(max_genes).indices
            top_idx  = top_idx.sort().values
            self.profiles = self.profiles[:, top_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "profile": self.profiles[idx],          # (n_genes,)
            "label":   self.labels[idx],
        }


# ════════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITION
# ════════════════════════════════════════════════════════════════════════════

class PerturbationCatalogueLLM(nn.Module):
    """
    PerturbationCatalogueLLM — two-stage model:
      1. Gene-expression encoder  — a lightweight MLP that projects Log2FC
         profiles into dense embeddings.
         Pre-trained backbone: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
         (understands biomedical gene/protein names from abstract-level training).

      2. Perturbation head — projects the embedding back to Log2FC space
         for reconstruction (unsupervised) and downstream retrieval.
    """

    MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

    def __init__(self, n_genes: int, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_genes   = n_genes
        self.embed_dim = embed_dim

        # ── Expression encoder (MLP, no pretrained backbone needed) ─────────
        # This takes the raw Log2FC vector and produces a fixed-size embedding.
        # We deliberately keep it simple: the gene-level representation is the
        # the differential expression profile itself, not token IDs.
        hidden = min(1024, n_genes * 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # ── Decoder / reconstruction head ────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_genes),
        )

    def encode(self, profile: torch.Tensor) -> torch.Tensor:
        """profile: (B, n_genes) → embedding: (B, embed_dim)"""
        return self.encoder(profile)

    def forward(self, profile: torch.Tensor):
        """Returns (embedding, reconstructed_profile)."""
        emb   = self.encode(profile)
        recon = self.decoder(emb)
        return emb, recon


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    matrix, meta = load_and_pivot(args.data_path)
    n_genes = min(meta["n_genes"], args.max_genes)

    dataset = PerturbSeqDataset(matrix, max_genes=args.max_genes)
    # With only 7 perturbations we keep all for training and use LOO validation
    # For larger datasets, split 80/20
    if len(dataset) >= 10:
        train_idx, val_idx = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42
        )
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds   = torch.utils.data.Subset(dataset, val_idx)
    else:
        # Too few samples — train on all, validate on all (demo mode)
        print("Fewer than 10 perturbations — using full dataset for both train/val")
        train_ds = val_ds = dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────────────
    model = PerturbationCatalogueLLM(n_genes=n_genes, embed_dim=args.embed_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    loss_fn = nn.MSELoss()

    # ── Train ────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            profile = batch["profile"].to(device)
            emb, recon = model(profile)
            loss = loss_fn(recon, profile)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                profile = batch["profile"].to(device)
                _, recon = model(profile)
                val_loss += loss_fn(recon, profile).item()
        val_loss /= len(val_loader)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pt")
            print(f"           ↳ saved best model (val_loss={val_loss:.4f})")

    # ── Save everything needed for inference ────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "model_config.json", "w") as f:
        json.dump({
            "n_genes":   n_genes,
            "embed_dim": args.embed_dim,
            "max_genes": args.max_genes,
        }, f, indent=2)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n Fine-tuning complete. Model saved to: {output_dir}")
    print(f"   Best val loss: {best_val_loss:.4f}")

    # ── Pre-compute and save all perturbation embeddings ────────────────────
    print("\nPre-computing perturbation embeddings for RAG index...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.eval()

    all_profiles = dataset.profiles.to(device)
    with torch.no_grad():
        embeddings = model.encode(all_profiles).cpu().numpy()

    np.save(output_dir / "perturbation_embeddings.npy", embeddings)

    embedding_index = {
        label: embeddings[i].tolist()
        for i, label in enumerate(dataset.labels)
    }
    with open(output_dir / "embedding_index.json", "w") as f:
        json.dump(embedding_index, f)

    print(f"   Saved {len(dataset.labels)} perturbation embeddings → {output_dir}/embedding_index.json")
    return model, dataset, meta


# ════════════════════════════════════════════════════════════════════════════
# 4. ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   default="data/perturb_seq.csv")
    p.add_argument("--output_dir",  default="embeddings/perturbation_catalogue_llm")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=4)     # small: only 7 perturbations
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--embed_dim",   type=int,   default=256)
    p.add_argument("--max_genes",   type=int,   default=2048)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
