# Phase 2 — Building a Grounded RAG Architecture

## Engineering Storybook

This document describes the second phase of the **GenAI Quality Architect** project.

Phase 1 established a resilient LLM service using FastAPI, HTTPX, Ollama, and a local Qwen3:4B model. Phase 2 extends that foundation with **Retrieval-Augmented Generation (RAG)**, so responses can be informed by document evidence rather than relying exclusively on the model's learned knowledge.

The objective was not simply to store PDFs in a vector database. It was to build a **testable, traceable, evidence-aware RAG pipeline**, with explicit handling for questions that lack sufficiently relevant retrieved context.

---

# 1. The Problem

A standalone LLM can produce fluent answers without having access to an organization's current policies or internal documents.

```text
Client
  |
  v
Chat API
  |
  v
LLM
  |
  v
Plausible response (not necessarily document-grounded)
```

This raises quality questions:

- Where did the answer come from?
- Is the answer supported by the provided document?
- Can the system identify the source page and chunk?
- What happens when the document does not contain the requested information?
- How can ingestion, retrieval, and generation be tested independently?
- Does a semantically similar passage actually contain enough evidence to answer?

The Phase 2 engineering goal was to **connect generation to retrievable evidence and make unsupported-answer behavior observable and testable**.

---

# 2. Architecture Decision

The Phase 2 architecture builds on the Phase 1 LLM service rather than replacing it.

```text
                     OFFLINE / PREPARATION

Policy PDF
    |
    v
PDF Loader (pages + source metadata)
    |
    v
Document Chunker
    |
    v
SentenceTransformers Embeddings
    |
    v
Persistent ChromaDB Collection

                     ONLINE / QUESTION ANSWERING

Client
    |
    v
FastAPI: POST /api/rag/ask
    |
    v
RAG Service
    |
    +--> Retriever --> Query Embedding --> ChromaDB Search
    |                                         |
    |                                         v
    +<---------------------------- Retrieved Chunks + Distances
    |
    v
EvidenceValidator (distance threshold)
    |
    +--> No relevant chunks --> Controlled abstention + [] sources
    |
    v
Context-constrained prompt
    |
    v
Existing LLM Service --> Ollama --> Qwen3:4B
    |
    v
Answer + source metadata
```

**Architecture principle:** Ingestion, retrieval, evidence filtering, and generation are separate components. This enables focused tests and clearer failure diagnosis.

---

# 3. Why Introduce RAG?

Without retrieval, a model may have no access to the policy document relevant to the question.

With RAG:

1. The document is loaded, chunked, embedded, and indexed.
2. A user question is converted into an embedding.
3. ChromaDB returns nearby document chunks.
4. An evidence filter removes chunks exceeding the configured cosine-distance threshold.
5. The remaining chunks are supplied to the LLM as context.
6. The API returns the generated answer and source metadata.

This design supports **traceability** and reduces dependence on unsupported model recall. It does **not** prove that every generated answer is faithful to its sources; that requires additional evaluation.

---

# 4. Technology Choices

Phase 2 continues the project's local, open-source-first approach.

| Component | Technology |
|---|---|
| Programming language | Python |
| API framework | FastAPI |
| API request validation | Pydantic |
| PDF parsing | pypdf |
| Document chunking | Custom Python chunker |
| Embedding model | SentenceTransformers `all-MiniLM-L6-v2` |
| Vector database | ChromaDB (persistent local collection) |
| Similarity strategy | Cosine distance |
| LLM runtime | Ollama |
| Local LLM | `qwen3:4b` |
| LLM communication | Existing async HTTPX service |
| Automated testing | Pytest, mocks, AsyncMock |

No paid model API is required for the demonstrated workflow.

---

# 5. Document Ingestion Pipeline

The ingestion script is `backend/ingest_documents.py`.

```text
backend/data/documents/policy_document.pdf
               |
               v
            load_pdf()
               |
               v
        chunk_documents()
               |
               v
     embed_documents(texts)
               |
               v
        VectorStore.add_chunks()
               |
               v
  backend/data/chroma/ (persistent)
```

The demonstrated ingestion run reported:

```text
Pages loaded: 4
Chunks created: 10
Embeddings generated: 10
Indexed chunks: 10
Document ingestion completed successfully.
```

Each chunk retains source metadata, including its document name, page, and chunk identifier. The ChromaDB files are generated runtime data and are excluded from Git tracking; the sample PDF and ingestion script can be versioned for reproducibility.

---

# 6. Chunking and Source Traceability

Long documents must be divided into retrievable units. The project uses configurable word-based chunking with overlap.

Example ingestion configuration:

```python
chunks = chunk_documents(
    pages,
    chunk_size=120,
    chunk_overlap=20,
)
```

Metadata accompanies each chunk:

```json
{
  "source": "policy_document.pdf",
  "page": 3,
  "chunk_id": "policy_document.pdf_p3_c1"
}
```

Chunk identifiers make retrieval results inspectable and support debugging when a passage is missed, split incorrectly, or retrieved unexpectedly.

**Tradeoff:** Smaller chunks can improve retrieval precision but may lose context; larger chunks preserve context but can introduce irrelevant text. The chosen settings are initial engineering defaults, not yet an optimized configuration.

---

# 7. Embeddings and Vector Storage

The `EmbeddingService` uses `all-MiniLM-L6-v2` to represent both document chunks and user queries as vectors.

```text
Document text ------> Document embedding --+
                                           |
                                           v
                                     ChromaDB index
                                           ^
                                           |
User question ------> Query embedding ------+
```

The `VectorStore` uses a persistent ChromaDB collection named `policy_documents`, configured for cosine-distance search.

The persistent collection is distinct from temporary databases created by integration tests. During development, the application collection initially contained **0 chunks** even though the retrieval tests had passed. Running the ingestion script populated it with **10 chunks**.

**Engineering lesson:** Passing an isolated integration test does not guarantee that the running application's persistent data store is populated.

---

# 8. Retriever Design

The `Retriever` separates query embedding and vector search from the API and generation layers.

Conceptual flow:

```text
Question
   |
   v
EmbeddingService.embed_query()
   |
   v
VectorStore.search(top_k=3)
   |
   v
[
  { text, metadata, distance },
  { text, metadata, distance },
  ...
]
```

The retrieval result contains:

- Chunk text
- Source metadata
- Cosine distance

A lower cosine distance generally indicates greater embedding similarity under this configuration. Similarity, however, is not equivalent to factual support or answerability.

---

# 9. Evidence Validation and Abstention

The `EvidenceValidator` filters retrieved chunks using a configurable maximum distance. The demonstrated configuration uses:

```python
EvidenceValidator(max_distance=0.65)
```

```text
Retrieved chunks
      |
      v
Distance filter
      |
      +--> No chunks pass --> Abstain; sources = []
      |
      v
Relevant chunks passed to prompt
```

This provides a first-line guard against answering from obviously unrelated passages.

**Important limitation:** A chunk passing the distance threshold is only *semantically similar*. The filter does not independently establish whether it contains sufficient evidence, whether the final answer is correct, or whether every returned source supports the answer. These distinctions motivate later evaluation phases.

---

# 10. Context-Grounded Generation

The `RAGService` orchestrates retrieval, evidence filtering, prompt construction, and the existing LLM service.

Its prompt directs the model to:

- Use only the supplied document context.
- Avoid inventing information.
- State that sufficient information was not found when the context cannot answer the question.

The implementation reuses the Phase 1 `generate_response()` service, retaining the existing Ollama integration and its timeout, retry, and logging behavior.

The RAG service returns an answer alongside the metadata of chunks supplied as context. These are **retrieved context references**, not independently verified, claim-level citations.

---

# 11. RAG API Contract

The new endpoint is:

```http
POST /api/rag/ask
```

Example request:

```json
{
  "question": "How frequently is production access reviewed?"
}
```

Observed response:

```json
{
  "answer": "Every 90 days.",
  "sources": [
    {
      "source": "policy_document.pdf",
      "page": 3,
      "chunk_id": "policy_document.pdf_p3_c1"
    },
    {
      "source": "policy_document.pdf",
      "page": 3,
      "chunk_id": "policy_document.pdf_p3_c2"
    }
  ]
}
```

The API is registered alongside the existing `/api/chat` endpoint. FastAPI lifespan initialization loads the embedding model, persistent vector store, retriever, and RAG service when the application starts.

A missing RAG service is handled as an initialization failure rather than silently producing an answer.

---

# 12. Failure Scenario — Unsupported Question

An important negative test asks for information not present in the indexed policy document.

```json
{
  "question": "What is the company's CEO's personal phone number?"
}
```

Observed response:

```json
{
  "answer": "I couldn't find sufficient information in the provided documents to answer this question.",
  "sources": []
}
```

In this case, retrieved chunks did not pass the evidence-distance filter, so the service **did not need to call the LLM**.

```text
Unsupported question
        |
        v
Semantic retrieval
        |
        v
Evidence filtering
        |
        v
No chunks accepted
        |
        v
Controlled abstention; no sources
```

This demonstrates the desired behavior for the tested scenario. It does not establish a universal hallucination-prevention guarantee.

---

# 13. The Testing Challenge

RAG introduces multiple failure locations that a single end-to-end assertion can obscure:

- **Ingestion:** Missing PDF, incorrect extraction, incomplete pages
- **Chunking:** Invalid size, overlap errors, loss of context
- **Embedding:** Unexpected shape, incorrect normalization, semantic mismatch
- **Storage:** Empty collection, persistence issues, incorrect metadata
- **Retrieval:** Irrelevant top results, invalid query, invalid `top_k`
- **Evidence filtering:** Threshold errors, unsupported questions accepted
- **Generation:** Context ignored, unsupported claims, missing attribution
- **API:** Invalid request, service not initialized, response-contract issues

The testing approach separates deterministic component behavior from real retrieval and model integration.

---

# 14. Quality Layers

```text
                       RAG QUALITY
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
       Component Tests  Integration    AI Evaluation
                         Tests         (later phases)
             |             |             |
             v             v             v
       Input validation  Real PDF      Retrieval metrics
       Chunking rules    Real model    Groundedness
       Metadata          ChromaDB      Faithfulness
       Thresholds        API flow      Answer correctness
       Mocked LLM        Local Ollama  Hallucination risk
```

Component tests can assert exact expected behavior. Real model output may vary and needs evaluation beyond simple exact-string matching.

The automated `RAGService` tests mock the LLM call, keeping service behavior deterministic without requiring Ollama for every test execution.

---

# 15. Automated Tests Implemented

The verified Phase 2 test coverage includes:

```text
Document Loader
   +-- Missing document
   +-- Real PDF extraction

Chunker
   +-- Valid chunking
   +-- Invalid chunk size
   +-- Invalid overlap

Embedding Service
   +-- Document embeddings
   +-- Query embedding
   +-- Relative semantic similarity

Vector Store
   +-- Add and search chunks

Retriever
   +-- Relevant chunk retrieval
   +-- Empty query rejected
   +-- Invalid top_k rejected

Integration
   +-- Embedding + vector retrieval
   +-- PDF retrieval pipeline

EvidenceValidator
   +-- Filter irrelevant chunks
   +-- Accept relevant evidence
   +-- Reject irrelevant evidence
   +-- Empty results
   +-- Invalid threshold

RAGService
   +-- Grounded-answer service path (mocked LLM)
   +-- Unsupported-question path (LLM not called)

Phase 1 Regression
   +-- Existing chat API tests
```

Verified automated results across the Day 2 checks:

| Test group | Passed |
|---|---:|
| Core document, embedding, vector and retrieval tests | 14 |
| Existing chat API + evidence validator tests | 10 |
| RAG service tests | 2 |
| **Total** | **26** |

Additionally, two live `/api/rag/ask` scenarios were exercised successfully: an answerable policy question and an unsupported personal-information question.

The test runs emitted dependency deprecation warnings, but no test failures. Automated endpoint-level tests for the new RAG route were not part of the reported 26-test total.

---

# 16. Key Learning

> **A successful API response does not prove that an AI answer is supported by evidence.**

Three distinctions matter:

```text
HTTP 200             != Correct answer
Relevant retrieval   != Sufficient evidence
Fluent response      != Grounded response
```

A document-grounded pipeline creates the *opportunity* to support claims with evidence. A retrieval threshold creates an initial relevance control. Neither alone validates the factual accuracy of the generated answer.

Another practical learning was the difference between **test infrastructure and application infrastructure**: the retrieval integration tests passed while the production-style persistent ChromaDB collection was empty. Explicit ingestion and startup verification closed that gap.

---

# 17. Phase 2 Result

Phase 2 established:

- A working PDF ingestion pipeline
- Page-aware chunking and traceable chunk metadata
- Local semantic embeddings
- Persistent ChromaDB storage
- Semantic retrieval with cosine-distance results
- A configurable evidence filter
- Controlled abstention when no retrieved chunks pass the filter
- Context-constrained prompting
- Reuse of the resilient Phase 1 LLM service
- A working `POST /api/rag/ask` endpoint
- Returned document/page/chunk references
- Positive and negative live API verification
- **26 passing automated tests**, including Phase 1 regression coverage

The resulting architecture is:

```text
                      APPLICATION

PDF --> Loader --> Chunker --> Embeddings --> ChromaDB
                                             ^
                                             |
Client --> FastAPI --> RAGService --> Retriever
                         |                   |
                         |<--- Chunks -------+
                         v
                  EvidenceValidator
                         |
                  +------+------+
                  |             |
               No match       Match
                  |             |
                  v             v
                Abstain    Context Prompt
                                |
                                v
                         LLM Service
                                |
                                v
                          Ollama / Qwen
                                |
                                v
                      Answer + Sources

                        QUALITY

        Pytest component + integration tests
                         |
         +---------------+---------------+
         |               |               |
         v               v               v
    Ingestion /      Retrieval /     RAG Service /
    Embeddings       Evidence         API behavior
```

**Phase 2 is complete for the demonstrated scope.** The next engineering challenge is not merely whether retrieval returns chunks, but **how to measure retrieval quality consistently**.

**Next — Phase 3:** Golden datasets, Hit Rate@K, Mean Reciprocal Rank (MRR), Recall@K, and measurable retrieval quality gates.
