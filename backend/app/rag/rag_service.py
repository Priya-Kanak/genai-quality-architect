from app.rag.evidence_validator import EvidenceValidator
from app.service.llm_service import generate_response

class RAGService:
    def __init__(self,retriever, max_distance = 0.65):
        self.retriever = retriever
        self.validator = EvidenceValidator(
            max_distance= max_distance
        )
        
    async def ask(self,question:str):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        results = self.retriever.retrieve(
            query = question,
            top_k = 3
        )
        
        relevant_chunks = (
            self.validator.filter_relevant_chunks(results)
        )
        
        if not relevant_chunks:
            return {
                "answer":(
                    "I couldn't find sufficient information "
                    "in the provided documents to answer "
                    "this question."
                ),
                "sources": [],
            }
        
        context_parts = []
        sources = []

        for chunk in relevant_chunks:
            metadata = chunk["metadata"]

            context_parts.append(
                f"Source: {metadata['source']}, "
                f"Page: {metadata['page']}\n"
                f"{chunk['text']}"
            )

            sources.append({
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
            })

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a policy document assistant.

Answer the question using ONLY the supplied context.
Do not invent information.

If the context does not contain the answer, respond:
"I couldn't find sufficient information in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        answer = await generate_response(prompt)

        return {
            "answer": answer.strip(),
            "sources": sources,
        }
