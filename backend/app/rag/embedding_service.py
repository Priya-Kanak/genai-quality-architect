from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingService:
    
    """Generate local embeddings for documents and queries."""
    
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for documents chunks.""" 
        
        embeddings= self.model.encode(
            texts,
            normalize_embeddings=True,
        )
        
        return embeddings.tolist()
    
    def embed_query(
        self,
        query:str,
    ) -> list[float]:
        
        """Generate an embedding for a user query.""" 
        
        embeddings = self.model.encode(
            query,
            normalize_embeddings= True,     
        )
        
        return embeddings.tolist()
     