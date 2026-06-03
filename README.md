# Project 3: Conversational RAG Agent — *The Gift of the Magi*

A production-quality conversational agent built with **PydanticAI** that answers questions about O. Henry's *The Gift of the Magi* using a full Retrieval-Augmented Generation (RAG) pipeline. The agent retrieves semantically relevant context from a **Qdrant Cloud** vector database, routes queries to specialised tools, and produces grounded, fact-only responses — with full observability via **LogFire**.

---

## Architecture Overview

```
User Input
    │
    ▼
ConversationOrchestrator  (project3.py)
    │
    ├── RAGPipeline         (RAG.py)          — Chunking, Embedding, Qdrant Cloud storage & retrieval
    ├── Dependencies        (dependencies.py) — Dataclass injecting shared clients into the agent
    └── Agent (PydanticAI)
            │
            ├── Tool: retrieve_rag_context    — Embeds query → searches Qdrant → returns top-k chunks
            ├── Tool: get_character_bio       — Key-value store for character definitions
            └── Tool: calculate_remaining_money — Deterministic financial arithmetic
```

The agent runs **statelessly** per turn — all shared infrastructure (OpenAI client, Qdrant client) is instantiated once at startup and injected via `Dependencies`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | [PydanticAI](https://ai.pydantic.dev/) |
| LLM | `gpt-4o-mini` via [aicredits.in](https://aicredits.in) (OpenAI-compatible) |
| Embeddings | `text-embedding-3-small` (1536-dim) |
| Vector Database | [Qdrant Cloud](https://qdrant.tech/) |
| Observability | [LogFire](https://logfire.pydantic.dev/) |
| Async Runtime | Python `asyncio` |

---

## Project Structure

```
project3/
├── project3.py         # Entry point — ConversationOrchestrator + REPL loop
├── RAG.py              # Full RAG pipeline: load → chunk → embed → store → retrieve
├── tools.py            # PydanticAI tool definitions used by the agent
├── dependencies.py     # Dependency injection dataclass
└── README.md
```

---

## RAG Pipeline

The ingestion pipeline (`RAG.py`) runs **once** to populate the Qdrant Cloud collection. Subsequent runs skip ingestion and connect directly to the existing collection.

**Steps:**

1. **Load** — Reads the raw Project Gutenberg `.txt` file from disk.
2. **Chunk** — Strips Gutenberg headers/footers, splits on paragraph boundaries with a **500-character window** and **100-character overlap**. Filters chunks shorter than 150 characters and deduplicates.
3. **Embed** — Batch-embeds all chunks via `text-embedding-3-small`. Each chunk is assigned a **deterministic UUID** derived from an MD5 hash of its text content, making upserts idempotent.
4. **Store** — Uploads `PointStruct` objects (id + vector + text payload) to Qdrant Cloud via `upsert`.
5. **Retrieve** — At query time, embeds the user's query and runs a cosine similarity search (`top_k=5`) against the collection.

To run ingestion manually:

```bash
python3 RAG.py
```

---

## Agent Tools

### `retrieve_rag_context(query: str)`
The primary tool. Embeds the user's query and retrieves the top-5 semantically matching chunks from Qdrant. The agent is instructed to call this whenever it needs story facts.

### `get_character_bio(character_name: str)`
A fast, deterministic key-value lookup for character definitions. Accepts `"della"`, `"jim"`, or `"sofronie"`. Intentionally separated from RAG to avoid retrieval noise on simple "who is X?" questions.

### `calculate_remaining_money(starting_amount: float, spent_amount: float)`
Handles all financial arithmetic deterministically in Python. The agent is explicitly instructed never to attempt the math itself.

---

## Observability (LogFire)

Every significant operation is wrapped in a LogFire span with structured attributes:

| Span | Key Attributes |
|---|---|
| `orchestrator.chat_turn` | `user_input` |
| `rag.load_file` | `file_characters` |
| `rag.chunking_process` | `total_chunks_generated` |
| `rag.generate_embeddings` | `batch_size`, `tokens_used` |
| `rag.store_in_qdrant` | `payload_size`, `uploaded_points` |
| `rag.retrieve_context` | `retrieved_chunks` |
| `Embedding user query` | `query_length`, `tokens_used` |

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/Dark5301/rag-agent-pydanticai.git
cd rag-agent-pydanticai
pip install pydantic-ai openai qdrant-client logfire python-dotenv
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
AICREDITS_API_KEY=your_aicredits_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Run ingestion (first time only)

```bash
python3 RAG.py
```

### 4. Start the agent

```bash
python3 project3.py
```

---

## Example Interaction

```
--- Project 3: The Gift of the Magi Orchestrator ---

Enter your question: How much money did Della have to start with?

Agent Response:
Della began with $1.87, which she had saved one cent at a time through careful bargaining with grocers and butchers.

Enter your question: Who is Madam Sofronie?

Agent Response:
Madam Sofronie is a large, chilly woman who runs a hair goods shop. She purchased Della's hair for twenty dollars.
```

---

## Design Decisions

- **Single event loop** — `asyncio.run()` is called exactly once in `__main__`, governing the entire application lifespan. No nested loops or repeated `asyncio.run()` calls.
- **Stateless agent runs** — The `Agent` instance is reused; `Dependencies` are packed fresh each turn. This keeps the agent cheap to run and easy to scale.
- **Deterministic chunk IDs** — MD5-based UUIDs ensure that re-running ingestion never creates duplicate points in Qdrant (upsert is safe to call multiple times).
- **Tool separation** — Character bios live in a key-value store, not RAG. This prevents the retrieval tool from being called for trivially answerable definitional questions.
- **Graceful RAG failure** — If Qdrant is unreachable, `chunk_retrieval` returns an empty string rather than raising. The agent can still respond, just without document context.
