# GenAI Quality Architect

An end-to-end **GenAI Quality Engineering and Test Architecture** project exploring how to design, test, evaluate, and determine the production readiness of modern AI applications.

The goal is not simply to build a chatbot rather focuses on the broader engineering question:

> **How do we build a quality architecture for an application containing APIs, microservices, RAG, LLMs, and AI agents?**

The project combines traditional Quality Engineering practices with AI-specific evaluation techniques such as retrieval evaluation, groundedness, faithfulness, hallucination detection, LLM-as-a-Judge, and agent trajectory evaluation.

---

## Project Objectives

This project aims to build a complete quality architecture covering:

- API and microservice testing
- LLM integration testing
- RAG pipeline testing
- Retrieval quality evaluation
- LLM response evaluation
- Hallucination and groundedness evaluation
- AI agent and tool-call testing
- Agent trajectory evaluation
- Performance and resilience testing
- Observability
- CI/CD quality gates
- Production-readiness assessment

The implementation primarily uses **free and open-source tools** and locally hosted models.

---

# Architecture

## Current Architecture — Phase 1

```text
                         CLIENT
                            |
                            v
                      +-----------+
                      |  FastAPI  |
                      +-----+-----+
                            |
                            v
                      +-----------+
                      | Chat API  |
                      +-----+-----+
                            |
                   Exception Mapping
                     503 / 504
                            |
                            v
                    +---------------+
                    |  LLM Service  |
                    +-------+-------+
                            |
                +-----------+-----------+
                |           |           |
             Timeout      Retry       Logging
                            |
                     Exponential
                       Backoff
                            |
                            v
                      +-----------+
                      |  Ollama   |
                      +-----+-----+
                            |
                            v
                      +-----------+
                      | Qwen 3 4B |
                      +-----------+
```

---

# Quality Architecture

```text
                         Pytest
                           |
              +------------+------------+
              |            |            |
              v            v            v
           Positive     Negative      Failure
             Tests        Tests        Tests
                                        |
                                  +-----+-----+
                                  |           |
                                  v           v
                                 503         504
```

The test architecture separates different quality concerns:

```text
Component/API Tests
        |
        |---- API contracts
        |---- Input validation
        |---- Error handling
        |---- Mocked dependencies
        |
Integration Tests
        |
        |---- FastAPI <-> Ollama
        |
AI Evaluation
        |
        |---- Retrieval quality
        |---- Answer correctness
        |---- Faithfulness
        |---- Groundedness
        |---- Hallucination
        |
Agent Evaluation
        |
        |---- Tool selection
        |---- Tool parameters
        |---- Agent trajectory
        |---- Task completion
        |
Production Quality
        |
        |---- Latency
        |---- Reliability
        |---- Cost
        |---- Observability
```

---

# Technology Stack

| Area | Technology |
|---|---|
| Backend | FastAPI |
| Language | Python |
| Local LLM Runtime | Ollama |
| LLM | Qwen |
| API Client | HTTPX |
| Data Validation | Pydantic |
| Test Framework | Pytest |
| Mocking | unittest.mock / AsyncMock |
| Vector Database | ChromaDB - planned |
| Embeddings | Sentence Transformers - planned |
| RAG Evaluation | RAGAS / DeepEval - planned |
| Agent Framework | LangGraph - planned |
| UI Automation | Playwright - planned |
| Performance Testing | k6 / Locust - planned |
| Observability | OpenTelemetry / Prometheus - planned |
| Dashboard | Grafana - planned |
| CI/CD | GitHub Actions - planned |

---

# Current Implementation

## Phase 1 — LLM Service Foundation

The first phase establishes a production-oriented foundation for integrating a local LLM.

Implemented capabilities:

- FastAPI application
- `/health` endpoint
- `/api/chat` endpoint
- Pydantic request/response contracts
- LLM service abstraction
- Local Ollama integration
- Qwen local model
- Async HTTP communication
- Timeout handling
- Connection failure handling
- Controlled retries
- Exponential backoff
- Custom LLM exceptions
- HTTP 503 mapping for LLM unavailability
- HTTP 504 mapping for LLM timeout
- Application logging
- Deterministic component testing through LLM mocking
- Positive API tests
- Negative API tests
- Failure-path tests

---

# Resilience Strategy

External AI infrastructure can fail temporarily.

The LLM service therefore distinguishes transient infrastructure failures from deterministic application failures.

```text
Request
   |
   v
Call LLM
   |
   v
Success? ------ YES ------> Response
   |
   NO
   |
   v
Classify Failure
   |
   v
Transient?
   / \
 YES   NO
  |     |
Retry  Fail Fast
  |
Backoff
  |
Max Attempts
  |
Fail Safely
```

Examples of potentially retryable failures include:

- Network interruption
- Connection failure
- Timeout
- Temporary service unavailability

Validation and deterministic client errors should generally fail fast rather than being blindly retried.

---

# Exception Architecture

Infrastructure-specific exceptions are translated into application-level exceptions.

```text
HTTPX / Ollama Failure
          |
          v
    LLM Service Layer
          |
          v
+--------------------------+
| LLMTimeoutError          |
| LLMUnavailableError      |
+------------+-------------+
             |
             v
          API Layer
             |
       +-----+-----+
       |           |
      504         503
```

This prevents API consumers from depending on internal implementation details.

---

# Testing Strategy

One of the central principles of this project is separating **deterministic software testing** from **probabilistic AI evaluation**.

## Traditional Software Testing

Traditional application behavior can often be validated using exact assertions.

```text
Expected Output == Actual Output
```

Examples:

- HTTP status codes
- API schema
- required fields
- authentication
- exception handling
- tool parameters
- service contracts

## GenAI Evaluation

LLM responses are probabilistic.

Two different responses may both be semantically correct.

Therefore:

```text
Quality(Response) >= Defined Threshold
```

AI outputs will eventually be evaluated using dimensions such as:

- Correctness
- Relevance
- Faithfulness
- Groundedness
- Completeness
- Hallucination
- Citation accuracy

---

# Automated Tests

Current automated tests cover:

```text
Health endpoint
      |
      +-- API health validation

Chat API
      |
      +-- Successful response
      |
      +-- Missing message
      |
      +-- Invalid message type
      |
      +-- LLM unavailable -> HTTP 503
      |
      +-- LLM timeout -> HTTP 504
```

Run the test suite using:

```bash
cd backend
pytest -v
```

---

# Running Locally

## Prerequisites

Install:

- Python
- Git
- Ollama

Verify Ollama:

```bash
ollama --version
```

Download the local model:

```bash
ollama pull qwen3:4b
```

---

## Create Python Environment

Navigate to the backend:

```bash
cd backend
```

Create the environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Start the Application

Ensure Ollama is running.

Then:

```bash
uvicorn app.main:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# API Examples

## Health

```http
GET /health
```

Expected response:

```json
{
  "status": "UP"
}
```

## Chat

```http
POST /api/chat
```

Request:

```json
{
  "message": "Explain risk-based testing."
}
```

Example response:

```json
{
  "response": "Generated response from the local LLM...",
  "model": "qwen3:4b"
}
```

---

# Project Roadmap

## Phase 1 — LLM Foundation

- [x] FastAPI
- [x] Local Ollama integration
- [x] Qwen model
- [x] LLM abstraction
- [x] API contracts
- [x] Retry strategy
- [x] Exponential backoff
- [x] Timeout handling
- [x] Exception handling
- [x] Logging
- [x] API tests
- [x] LLM mocking
- [x] Failure-path testing

## Phase 2 — RAG Architecture

- [ ] Document ingestion
- [ ] PDF processing
- [ ] Chunking strategy
- [ ] Local embeddings
- [ ] ChromaDB
- [ ] Semantic retrieval
- [ ] Top-K retrieval
- [ ] Grounded generation
- [ ] Source citations

## Phase 3 — RAG Evaluation

- [ ] Golden evaluation dataset
- [ ] Hit Rate@K
- [ ] Precision@K
- [ ] Recall@K
- [ ] Mean Reciprocal Rank (MRR)
- [ ] Context relevance
- [ ] Retrieval failure analysis

## Phase 4 — LLM Evaluation

- [ ] Answer correctness
- [ ] Answer relevance
- [ ] Faithfulness
- [ ] Groundedness
- [ ] Hallucination detection
- [ ] RAGAS
- [ ] DeepEval
- [ ] LLM-as-a-Judge

## Phase 5 — Agentic AI

- [ ] LangGraph agents
- [ ] Tool calling
- [ ] Multi-step workflows
- [ ] Agent trajectory capture
- [ ] Tool-selection evaluation
- [ ] Tool-parameter validation
- [ ] Loop detection
- [ ] Agent failure recovery
- [ ] Task-success evaluation

## Phase 6 — End-to-End Quality Engineering

- [ ] React UI
- [ ] Playwright automation
- [ ] API integration testing
- [ ] Contract testing
- [ ] Performance testing
- [ ] Security testing
- [ ] Failure injection

## Phase 7 — Production Readiness

- [ ] GitHub Actions pipeline
- [ ] Automated AI evaluation
- [ ] Quality thresholds
- [ ] Release quality gates
- [ ] OpenTelemetry
- [ ] Prometheus
- [ ] Grafana
- [ ] Production monitoring
- [ ] Production-readiness dashboard

---

# Target Architecture

The final architecture will evolve toward:

```text
                         User
                           |
                           v
                       React UI
                           |
                           v
                       FastAPI
                           |
                           v
                      Orchestrator
                      /          \
                     /            \
                    v              v
              RAG Service     Agent Service
                    |              |
                    v              v
                Retriever       LangGraph
                    |              |
                    v              v
                ChromaDB          Tools
                    |              |
                    +------+-------+
                           |
                           v
                         Ollama
                           |
                           v
                      Local LLM


              QUALITY ENGINEERING PLATFORM

       +----------------+----------------+
       |                |                |
       v                v                v
   Playwright          Pytest       AI Evaluation
       |                |                |
       v                v                v
    UI/E2E             API        Golden Dataset
                                     |
                           +---------+---------+
                           |         |         |
                           v         v         v
                       Retrieval   LLM      Agent
                         Eval      Eval      Eval
                           \         |         /
                            \        |        /
                             +-------+-------+
                                     |
                                     v
                                Quality Gate
                                     |
                              +------+------+
                              |             |
                             PASS          FAIL
                              |             |
                              v             v
                           Deploy      Block Release
```

---

# Key Engineering Principle

A wrong GenAI response should not immediately be classified as an "LLM failure."

For a RAG system:

```text
                     Wrong Answer
                          |
                Was correct context
                    retrieved?
                    /        \
                  NO          YES
                  |            |
                  v            v
             Retrieval      Generation
              Failure        Failure
```

The quality architecture should therefore evaluate each AI subsystem independently before evaluating the final end-to-end response.

---

# Why This Project?

Modern Quality Engineering is expanding beyond UI and API automation.

AI-enabled applications introduce new quality dimensions:

- Non-deterministic outputs
- Retrieval quality
- Hallucinations
- Prompt sensitivity
- Model variability
- Agent decision making
- Tool usage
- Safety
- Latency
- Cost
- Observability

This project explores how traditional test architecture can evolve to address these challenges while maintaining measurable and automated release criteria.

---

## Status

**Phase 1: Complete**

Current focus:

**Phase 2 — RAG Architecture and Retrieval Evaluation**

---

## Author

**Priya Kanak**

Quality Engineering | Test Architecture | Test Automation | GenAI Quality Engineering
