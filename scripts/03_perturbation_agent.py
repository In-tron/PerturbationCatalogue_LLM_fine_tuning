"""
03_perturbation_agent.py
------------------------
LLM agent that answers biology questions by:
  1. Parsing the user's question to identify perturbation gene(s)
  2. Retrieving relevant PerturbationCatalogueLLM embeddings + Perturb-seq text from the RAG index
  3. Injecting that context into the LLM prompt
  4. Returning a grounded, citation-backed answer

Supports three LLM backends (swap via --llm flag):
  - openai    : GPT-4o / GPT-4  (requires OPENAI_API_KEY)
  - anthropic : Claude Sonnet/Opus (requires ANTHROPIC_API_KEY)
  - ollama    : Local Llama3 / Mistral (requires Ollama running locally)

USAGE:
    # Interactive mode
    python 03_perturbation_agent.py --llm anthropic

    # Single question
    python 03_perturbation_agent.py \
        --llm openai \
        --question "What happens when SPI1 is knocked out?"

    # Start as API server
    python 03_perturbation_agent.py --llm ollama --serve
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Import RAG index from step 02 ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.s02_build_rag_index import PerturbationRAGIndex  # type: ignore  # noqa


# ════════════════════════════════════════════════════════════════════════════
# 1. LLM BACKENDS
# ════════════════════════════════════════════════════════════════════════════

class LLMBackend:
    """Base class — all backends implement .complete(messages) → str."""
    def complete(self, messages: list[dict]) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        print(f"  LLM backend: OpenAI ({model})")

    def complete(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-5"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY environment variable")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model  = model
        print(f"  LLM backend: Anthropic ({model})")

    def complete(self, messages: list[dict]) -> str:
        # Separate system message from user/assistant turns
        system_msg = ""
        turns = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                turns.append(m)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_msg,
            messages=turns,
        )
        return response.content[0].text


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.1:8b"):
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("pip install requests")
        self.model   = model
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  LLM backend: Ollama ({model}) at {self.base_url}")

    def complete(self, messages: list[dict]) -> str:
        response = self.requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_backend(name: str, model: str = None) -> LLMBackend:
    backends = {
        "openai":    (OpenAIBackend,    model or "gpt-4o"),
        "anthropic": (AnthropicBackend, model or "claude-sonnet-4-5"),
        "ollama":    (OllamaBackend,    model or "llama3.1:8b"),
    }
    if name not in backends:
        raise ValueError(f"Unknown backend '{name}'. Choose: {list(backends)}")
    cls, default_model = backends[name]
    return cls(default_model)


# ════════════════════════════════════════════════════════════════════════════
# 2. AGENT
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a computational biology assistant specialising in
perturbation genomics. You answer questions about how genetic knockouts and
perturbations affect gene expression in cells.

You have access to a Perturb-seq database (Adamson 2016 pilot dataset) that
contains differential expression results for {n_perturbations} perturbations:
{perturbation_list}

When answering:
1. Base your answer PRIMARILY on the retrieved experimental data provided in
   the context — do not confabulate gene names or fold-changes.
2. Cite specific genes and Log2FC values from the context.
3. If the question asks about a perturbation NOT in the database, say so clearly
   and offer what general knowledge you have.
4. Use clear biological language — explain what up/down-regulation means
   functionally where relevant.
5. Structure longer answers with: Summary → Key effects → Mechanistic insight.
"""


class PerturbationAgent:
    """
    RAG-powered agent for Perturb-seq question answering.
    """

    def __init__(
        self,
        rag_index: PerturbationRAGIndex,
        llm:       LLMBackend,
        k_retrieve: int = 3,
    ):
        self.rag   = rag_index
        self.llm   = llm
        self.k     = k_retrieve
        self.history: list[dict] = []   # conversation memory

        self._system = SYSTEM_PROMPT.format(
            n_perturbations = len(rag_index.get_all_perturbations()),
            perturbation_list = ", ".join(rag_index.get_all_perturbations()),
        )

    def _extract_gene_names(self, question: str) -> list[str]:
        """
        Heuristic: extract candidate gene names from the question.
        Gene names are typically ALL-CAPS, 2–10 chars, may contain digits.
        """
        import re
        # Common gene-name pattern
        candidates = re.findall(r'\b[A-Z][A-Z0-9]{1,9}\b', question)
        known = {p.upper() for p in self.rag.get_all_perturbations()}
        matched = [c for c in candidates if c in known]
        return matched

    def _retrieve_context(self, question: str) -> tuple[str, list[str]]:
        """
        Returns (context_text, source_perturbations).
        Strategy:
          1. Try direct gene-name match first (exact lookup)
          2. Fall back to embedding-based similarity on average of known embeddings
             (simple heuristic — could be replaced with a query encoder)
        """
        # Direct match
        gene_names = self._extract_gene_names(question)
        retrieved_docs = []
        sources = []

        for gene in gene_names:
            doc = self.rag.query_by_name(gene)
            if doc:
                retrieved_docs.append(doc)
                sources.append(gene)

        # If no direct match, retrieve by similarity to all embeddings
        # (use the mean embedding as a proxy for "general" query)
        if not retrieved_docs:
            all_embs = self.rag.embeddings
            mean_emb = all_embs.mean(axis=0)
            results  = self.rag.query_by_vector(mean_emb, k=self.k)
            for r in results:
                retrieved_docs.append(r["document"])
                sources.append(r["perturbation"])

        # Combine into a single context block
        context_parts = []
        for doc in retrieved_docs[:self.k]:
            context_parts.append(
                f"=== PERTURB-SEQ DATA: {doc['perturbation']} KNOCKOUT ===\n"
                + doc["text"]
            )

        context = "\n\n".join(context_parts)
        return context, sources

    def ask(self, question: str, stream_callback=None) -> str:
        """
        Ask a question. Returns the agent's answer as a string.
        Maintains conversation history for multi-turn dialogue.
        """
        # Retrieve relevant context
        context, sources = self._retrieve_context(question)

        # Build the augmented user message
        augmented_question = f"""RETRIEVED PERTURB-SEQ CONTEXT:
{context}

---

USER QUESTION:
{question}

Please answer based on the experimental data above. Cite specific genes and 
statistics where relevant. Sources used: {', '.join(sources) if sources else 'general retrieval'}"""

        # Build message list (with history for multi-turn)
        messages = [{"role": "system", "content": self._system}]
        messages += self.history[-6:]   # last 3 turns
        messages.append({"role": "user", "content": augmented_question})

        # Call LLM
        answer = self.llm.complete(messages)

        # Update history with clean (non-augmented) question
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

        return answer

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def structured_query(self, perturbation: str, question_type: str) -> dict:
        """
        Structured query API for programmatic use.

        question_type options:
          - 'top_effects'   : What are the strongest effects?
          - 'upregulated'   : Which genes are most up-regulated?
          - 'downregulated' : Which genes are most down-regulated?
          - 'mechanism'     : What is the likely biological mechanism?
          - 'comparison'    : Compare to other perturbations in the database
        """
        question_templates = {
            "top_effects":   f"What are the strongest expression effects of knocking out {perturbation}?",
            "upregulated":   f"Which genes are most up-regulated when {perturbation} is knocked out, and what do they suggest biologically?",
            "downregulated": f"Which genes are most down-regulated when {perturbation} is knocked out?",
            "mechanism":     f"Based on the perturb-seq data, what is the likely biological role of {perturbation}?",
            "comparison":    f"How does knocking out {perturbation} compare to the other perturbations in the dataset?",
        }
        question = question_templates.get(
            question_type,
            f"Tell me about the effects of knocking out {perturbation}."
        )
        answer = self.ask(question)
        doc    = self.rag.query_by_name(perturbation)

        return {
            "perturbation": perturbation,
            "question":     question,
            "answer":       answer,
            "metadata":     doc["metadata"] if doc else {},
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. FASTAPI SERVER (optional)
# ════════════════════════════════════════════════════════════════════════════

def make_app(agent: PerturbationAgent):
    """
    Create a FastAPI app wrapping the agent.
    Run with: uvicorn 03_perturbation_agent:app --reload
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    app = FastAPI(
        title="Perturbation RAG Agent",
        description="Ask questions about Perturb-seq data",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class QuestionRequest(BaseModel):
        question: str
        reset_history: bool = False

    class StructuredRequest(BaseModel):
        perturbation: str
        question_type: str = "top_effects"

    @app.get("/perturbations")
    def list_perturbations():
        return {"perturbations": agent.rag.get_all_perturbations()}

    @app.post("/ask")
    def ask(req: QuestionRequest):
        if req.reset_history:
            agent.reset()
        answer = agent.ask(req.question)
        return {"answer": answer}

    @app.post("/structured")
    def structured(req: StructuredRequest):
        return agent.structured_query(req.perturbation, req.question_type)

    @app.get("/health")
    def health():
        return {"status": "ok", "n_perturbations": len(agent.rag.get_all_perturbations())}

    return app


# ════════════════════════════════════════════════════════════════════════════
# 4. INTERACTIVE CLI
# ════════════════════════════════════════════════════════════════════════════

def interactive_loop(agent: PerturbationAgent):
    print("\n" + "="*60)
    print("Perturbation RAG Agent — Interactive Mode")
    print(f"Available perturbations: {', '.join(agent.rag.get_all_perturbations())}")
    print("Commands: 'reset' | 'quit' | or ask any biology question")
    print("="*60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "reset":
            agent.reset()
            print("  [History cleared]")
            continue

        print("\nAgent: ", end="", flush=True)
        answer = agent.ask(question)
        print(answer)
        print()


# ════════════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    print("Loading RAG index...")
    rag = PerturbationRAGIndex.load(args.index_dir)

    print("Initialising LLM backend...")
    llm = get_backend(args.llm, args.model)

    agent = PerturbationAgent(rag, llm, k_retrieve=args.k)

    if args.serve:
        app = make_app(agent)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    elif args.question:
        answer = agent.ask(args.question)
        print("\nAnswer:\n", answer)

    else:
        interactive_loop(agent)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", default="embeddings/rag_index")
    p.add_argument("--llm",       default="anthropic",
                   choices=["openai", "anthropic", "ollama"])
    p.add_argument("--model",     default=None,
                   help="Override default model for the chosen backend")
    p.add_argument("--k",         type=int, default=3,
                   help="Number of documents to retrieve")
    p.add_argument("--question",  default=None,
                   help="Single question mode")
    p.add_argument("--serve",     action="store_true",
                   help="Start as FastAPI server")
    p.add_argument("--port",      type=int, default=8000)
    main(p.parse_args())
