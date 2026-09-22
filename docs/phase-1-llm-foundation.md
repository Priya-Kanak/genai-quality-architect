# Phase 1 — Building a Resilient LLM Service

## Engineering Storybook

This document describes the first phase of the **GenAI Quality Architect** project.

The objective of Phase 1 was not simply to connect an application to an LLM.

The objective was to establish a **testable, resilient, and loosely coupled foundation** upon which RAG, AI agents, evaluation frameworks, and production quality gates can later be built.

---

# 1. The Problem

A basic GenAI application can be implemented as:

```text
Client
  |
  v
API
  |
  v
LLM
```

This works for a prototype, but it creates several quality and architectural questions.

What happens if:

- The LLM becomes unavailable?
- The request times out?
- A temporary network failure occurs?
- The model provider changes?
- The LLM generates a different answer every time?
- Automated tests require the real model to be running?
- Internal infrastructure exceptions leak through the public API?

The first phase therefore focused on building the LLM integration as an engineering component rather than treating the model as a simple API call.

---

# 2. Architecture Decision

The initial architecture was designed as:

```text
                         Client
                            |
                            v
                       FastAPI
                            |
                            v
                        Chat API
                            |
                            v
                      LLM Service
                            |
                            v
                         Ollama
                            |
                            v
                       Qwen 3 4B
```

A dedicated `LLM Service` was introduced between the API and the model runtime.

The API therefore does not directly communicate with Ollama.

---

# 3. Why Introduce an LLM Service?

Without abstraction:

```text
Chat API
   |
   v
Ollama-specific HTTP logic
```

The API becomes tightly coupled to the model infrastructure.

Instead:

```text
Chat API
   |
   v
LLM Service
   |
   +------ Ollama
   |
   +------ Future Model Provider
```

The service layer owns:

- Model communication
- Timeouts
- Retries
- Error classification
- Logging
- Provider-specific behavior

The API layer owns:

- HTTP contracts
- Input validation
- Response contracts
- HTTP error mapping

This separation improves:

- Testability
- Maintainability
- Replaceability
- Failure isolation
- Separation of concerns

---

# 4. Technology Choices

Phase 1 intentionally uses free and locally executable technologies.

| Component | Technology |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| Validation | Pydantic |
| LLM Runtime | Ollama |
| LLM | Qwen 3 4B |
| HTTP Client | HTTPX |
| Testing | Pytest |
| Mocking | AsyncMock |
| Logging | Python logging |

No paid LLM API is required.

---

# 5. API Contract

The first GenAI endpoint is:

```http
POST /api/chat
```

Example request:

```json
{
  "message": "Explain risk-based testing."
}
```

Example response:

```json
{
  "response": "Generated response...",
  "model": "qwen3:4b"
}
```

Pydantic is used to enforce the request and response contracts.

This allows invalid input to be rejected before reaching the LLM layer.

---

# 6. Failure Scenario — LLM Unavailable

A real AI system must assume external dependencies can fail.

Example:

```text
Client
  |
  v
FastAPI
  |
  v
LLM Service
  |
  X
Ollama unavailable
```

The application should not expose low-level errors such as:

```text
ConnectionRefusedError
httpx.ConnectError
```

Instead, infrastructure failures are translated into application-level exceptions.

```text
httpx.ConnectError
        |
        v
LLMUnavailableError
        |
        v
Chat API
        |
        v
HTTP 503
```

The API consumer receives a controlled response:

```json
{
  "detail": "LLM service unavailable"
}
```

---

# 7. Failure Scenario — LLM Timeout

Another important scenario is model latency.

```text
Request
   |
   v
LLM Service
   |
   v
Wait for LLM
   |
   X
Timeout
```

The failure is translated as:

```text
httpx.TimeoutException
        |
        v
LLMTimeoutError
        |
        v
Chat API
        |
        v
HTTP 504
```

This prevents infrastructure-specific exceptions from leaking into the API contract.

---

# 8. Retry Strategy

Not every failure should immediately fail the request.

Temporary infrastructure problems may recover.

The service therefore introduces controlled retry behavior.

```text
Request
   |
   v
Attempt 1
   |
   X
Failure
   |
   v
Wait
   |
   v
Attempt 2
   |
   X
Failure
   |
   v
Wait
   |
   v
Attempt 3
   |
   X
Failure
   |
   v
Fail Safely
```

Phase 1 uses exponential backoff:

```text
Attempt 1 -> 1 second
Attempt 2 -> 2 seconds
Attempt 3 -> 4 seconds
```

Conceptually:

```python
delay = 2 ** attempt
```

---

# 9. Retry Classification

Retries should not be applied blindly.

Potentially retryable failures include:

```text
Timeout
Connection failure
Temporary network failure
503 Service Unavailable
429 Too Many Requests
```

Failures that generally should not be blindly retried include:

```text
400 Bad Request
401 Unauthorized
403 Forbidden
Schema validation failure
Invalid input
Business assertion failure
```

This distinction prevents unnecessary traffic and hides fewer genuine defects.

---

# 10. Observability

Logging was introduced at the LLM service boundary.

Example execution:

```text
Calling LLM model=qwen3:4b attempt=1
LLM unavailable attempt=1

Calling LLM model=qwen3:4b attempt=2
LLM unavailable attempt=2

Calling LLM model=qwen3:4b attempt=3
LLM unavailable attempt=3
```

This is only the beginning of the observability architecture.

Future phases will introduce:

- Request IDs
- Distributed tracing
- Latency metrics
- Token/model metrics
- OpenTelemetry
- Prometheus
- Grafana

---

# 11. The Testing Challenge

LLMs introduce an important testing problem.

A question such as:

```text
What is risk-based testing?
```

could produce:

```text
Risk-based testing prioritizes testing according to business and technical risk.
```

Another execution might produce:

```text
Risk-based testing focuses testing effort on areas with the highest potential impact.
```

Both answers may be valid.

An assertion such as:

```python
assert actual_response == expected_response
```

would therefore be inappropriate for evaluating semantic quality.

---

# 12. Deterministic Testing vs AI Evaluation

This leads to one of the core architectural principles of the project.

## Traditional software behavior

```text
Input
  |
  v
Deterministic Logic
  |
  v
Expected Output

Expected == Actual
```

Examples include:

- HTTP status codes
- API schemas
- Required fields
- Error mapping
- Authentication
- Service contracts

## AI behavior

```text
Input
  |
  v
Probabilistic Model
  |
  v
Variable Response

Quality(Response) >= Defined Threshold
```

AI responses therefore require evaluation dimensions such as:

- Correctness
- Relevance
- Groundedness
- Faithfulness
- Hallucination
- Completeness

These will be implemented in later phases.

---

# 13. Why Mock the LLM?

Component tests should not require the real model to run.

Otherwise tests become:

- Slow
- Environment dependent
- Non-deterministic
- Difficult to reproduce
- More prone to flaky behavior

The LLM dependency is therefore mocked during API component testing.

```text
API Test
   |
   v
FastAPI
   |
   v
Mocked LLM
```

This allows deterministic verification of application behavior.

Real LLM behavior will be tested separately through integration tests and AI evaluation suites.

---

# 14. Quality Layers

Phase 1 established an important separation between different types of testing.

```text
                    Quality Architecture

                           |
          +----------------+----------------+
          |                |                |
          v                v                v

     Component Tests   Integration Tests   AI Evaluation

          |                |                |
          v                v                v

     API behavior      Real Ollama        Semantic
     Validation        communication      quality
     Error mapping                        Groundedness
     Mocking                              Hallucination
```

A single test suite should not attempt to answer all three questions.

---

# 15. Automated Tests Implemented

Phase 1 currently covers:

```text
Health API
   |
   +-- Service health

Chat API
   |
   +-- Successful response
   |
   +-- Missing message
   |
   +-- Invalid message type
   |
   +-- LLM unavailable
   |      |
   |      +-- HTTP 503
   |
   +-- LLM timeout
          |
          +-- HTTP 504
```

All Phase 1 automated tests are passing.

---

# 16. Key Learning

The most important lesson from Phase 1 is:

> Testing a GenAI application requires separating deterministic software behavior from probabilistic AI behavior.

The API contract, HTTP status codes, retries, exception handling, schemas, and service interactions can be tested deterministically.

The quality of an LLM-generated answer cannot always be evaluated using exact string comparison.

This distinction becomes the foundation for the evaluation architecture implemented in subsequent phases.

---

# 17. Phase 1 Result

Phase 1 established:

- A working local LLM
- A FastAPI application
- An LLM abstraction layer
- Typed API contracts
- Retry handling
- Exponential backoff
- Timeout handling
- Failure classification
- Exception translation
- Logging
- Dependency mocking
- Positive testing
- Negative testing
- Failure-path testing

The resulting foundation is:

```text
                       APPLICATION

Client
  |
  v
FastAPI
  |
  v
Chat API
  |
  v
LLM Service
  |
  +---- Timeout
  |
  +---- Retry
  |
  +---- Error handling
  |
  +---- Logging
  |
  v
Ollama
  |
  v
Qwen


                         QUALITY

                          Pytest
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
           Positive      Negative       Failure
             Tests         Tests          Tests
```
