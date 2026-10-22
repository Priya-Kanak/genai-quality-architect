class Retriever:
    """Retrieve relevant document chunks using semantic search."""
    
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
    def retrieve(self, query: str, top_k, int=3) -> list[dict]:
        """Retieve the most relevant chunks."""
        
        if not query.strip():
            raise ValueError(" Query cannot be empty")
        
        if top_k < 1 :
            raise ValueError("top_k must be atleast 1")
        
        if hasattr(self.embedding_model, "embed_query"):
            query_embedding = self.embedding_model.embed_query(query)
        else:
            query_embedding = self.embedding_model.encode(query)
        
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()
            
        results = self.vector_store.search(
            query_embedding = query_embedding,
            top_k = top_k,
        )
        
        retrieved_chunks = []
        
        for document, metadata, distance in zip (
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved_chunks.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance":distance
                }
            )
        return retrieved_chunks
        
        