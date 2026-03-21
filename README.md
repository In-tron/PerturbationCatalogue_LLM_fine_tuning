# PerturbationCatalogueLLM

LLM agent that answers biology questions grounded in your Perturb-seq data,
using fine-tuned PerturbationCatalogueLLM embeddings for retrieval.

## Architecture

```
Perturb-seq CSV  (Log2FC, Padj per perturbation × gene)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 1: Fine-tune PerturbationCatalogueLLM │
│  Pre-trained: BiomedNLP-BiomedBERT          │
│  MLP: Log2FC profile → 256-dim embedding    │
│  Trained with reconstruction loss           │
│  Output: embedding_index.json               │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 2: Build RAG index                    │
│  FAISS vector store  +  text documents      │
│  Text = natural-language perturb-seq summary│
│  Output: embeddings/rag_index/              │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 3: LLM Agent                          │
│  User question → extract gene names         │
│                → retrieve top-k documents   │
│                → augment prompt             │
│                → LLM generates answer       │
│  Backends: Claude (Anthropic) / GPT-4       │
│            (OpenAI) / Llama3 (Ollama local) │
└─────────────────────────────────────────────┘
```

## Quick Start (Colab)

1. Upload your CSV to Google Drive
2. Open `notebooks/pipeline.ipynb`
3. Set your API key in cell 2
4. Run all cells

## Quick Start (local)

```bash
pip install -r requirements.txt

# Copy your data
cp perturb-seq_adamson_2016_pilot.csv data/perturb_seq.csv

# Step 1: Fine-tune PerturbationCatalogueLLM
python scripts/01_finetune.py \
    --data_path data/perturb_seq.csv \
    --output_dir embeddings/perturbation_catalogue_llm \
    --epochs 100

# Step 2: Build index
python scripts/02_build_rag_index.py \
    --data_path data/perturb_seq.csv \
    --embeddings_dir embeddings/perturbation_catalogue_llm \
    --output_dir embeddings/rag_index

# Step 3: Ask questions
export ANTHROPIC_API_KEY=your-key-here
python scripts/03_perturbation_agent.py --llm anthropic

# Or start as API server
python scripts/03_perturbation_agent.py --llm anthropic --serve
```

## LLM Backends

| Backend | Flag | Env var | Notes |
|---|---|---|---|
| Claude (Anthropic) | `--llm anthropic` | `ANTHROPIC_API_KEY` | Default: claude-sonnet-4-5 |
| GPT-4o (OpenAI) | `--llm openai` | `OPENAI_API_KEY` | Default: gpt-4o |
| Llama3 (local) | `--llm ollama` | — | Requires Ollama running locally |

## API Endpoints (when using --serve)

```
GET  /perturbations          List all perturbation genes in the index
POST /ask                    Free-text question  → answer
POST /structured             Structured query    → answer + metadata
GET  /health                 Status check
```

Example POST /ask:
```json
{"question": "What happens when SPI1 is knocked out?"}
```

Example POST /structured:
```json
{"perturbation": "DDIT3", "question_type": "mechanism"}
```
question_type options: top_effects | upregulated | downregulated | mechanism | comparison

## File Structure

```
perturbation_rag/
├── data/
│   └── perturb_seq.csv              ← your Perturb-seq data
├── embeddings/
│   ├── perturbation_catalogue_llm/
│   │   ├── best_model.pt            ← fine-tuned PerturbationCatalogueLLM weights
│   │   ├── embedding_index.json     ← {gene: [embedding vector]}
│   │   ├── meta.json                ← gene list, perturbation list
│   │   └── model_config.json
│   └── rag_index/
│       ├── faiss.index              ← FAISS vector index
│       ├── documents.json           ← text summaries per perturbation
│       ├── embeddings.npy
│       └── labels.json
├── scripts/
│   ├── 01_finetune.py          ← fine-tunes PerturbationCatalogueLLM
│   ├── 02_build_rag_index.py
│   └── 03_perturbation_agent.py
├── notebooks/
│   └── pipeline.ipynb
└── requirements.txt
```

## Extending to More Data

- **More perturbations**: add rows to the CSV and re-run steps 1–2
- **scPerturb-seq Atlas**: same format works — just ensure Log2FC/Padj columns
- **MAVE data**: add a variant column; extend the text document builder
- **CRISPR screens**: aggregate by gene, compute pseudo-Log2FC from LFC scores
