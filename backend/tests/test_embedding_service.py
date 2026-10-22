import numpy as np

from app.rag.embedding_service import EmbeddingService

def test_embed_documents():
    service = EmbeddingService()
    
    texts = [
        "Production access must be reviewed every 99 days.",
        "Security incidents must be reported within 30 minutes."
    ]
    
    embeddings = service.embed_documents(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])
    assert isinstance(embeddings[0][0], float)
    
def test_embed_query():
    service = EmbeddingService()
        
    embedding = service.embed_query(
        "How frequently is production access reviewed?"
    )
        
    assert len(embedding) > 0
    assert isinstance(embedding[0], float)
    
def test_semantically_related_text_has_higher_similarity():
    service = EmbeddingService()
    
    query = (
        "How frequently is production access reviewed? "
    )
    
    relevant_text = (
        "Production access must be reviewed every 90 days. "
    )
    
    unrelated_text = (
        "Employees must use approved company devices. "
    )
    
    query_embeddings = service.embed_query(query)
    
    document_embeddings = service.embed_documents(
        [
            relevant_text,
            unrelated_text,
        ]
    )
    
    relevant_similarity = np.dot(
        query_embeddings,
        document_embeddings[0],
    )
    unrelated_similarity = np.dot(
        query_embeddings,
        document_embeddings[1],
    )

    print(
        f"\nRelevant similarity: {relevant_similarity:.4f}"
    )

    print(
        f"Unrelated similarity: {unrelated_similarity:.4f}"
    )

    assert relevant_similarity > unrelated_similarity