from sentence_transformers import SentenceTransformer

from app.rag.vector_store import VectorStore
from app.rag.retriever import Retriever

def test_real_embedding_and_vector_retrieval(tmp_path):
    #Load the real embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    
    #create isolated chroma db instance
    store = VectorStore(
        persist_directory= str(tmp_path),
        collection_name="integration_test",
    )
    
    
    chunks = [
        {
            "text": "Production access is reviewed every 90 days.",
            "metadata": {
                "source": "policy.pdf",
                "page": 1,
                "chunk_id": "policy_p1_c1",
                "chunk_number": 1,
            },
        },
        {
            "text": "Lost devices must be reported within one hour.",
            "metadata": {
                "source": "policy.pdf",
                "page": 2,
                "chunk_id": "policy_p2_c1",
                "chunk_number": 1,
            },
        },
        {
            "text": "Employees must complete security training annually.",
            "metadata": {
                "source": "policy.pdf",
                "page": 3,
                "chunk_id": "policy_p3_c1",
                "chunk_number": 1,
            },
        },
    ]
    
    #Generate real document embeddings
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks]
    ).tolist()
    
    #Store document chunks and embedding
    store.add_chunks(chunks, embeddings)
    
    #Create real retriever
    retriever = Retriever(
        embedding_model=model,
        vector_store=store,
    )
    
    #Retrieve real evidence
    results = retriever.retrieve(
        query="How frequently is production access is reviewed?",
        top_k=1,
    )
    
    #Validate Retrieval quality
    assert len(results) == 1

    assert "90 days" in results[0]["text"]

    # Validate source provenance
    assert results[0]["metadata"]["source"] == "policy.pdf"
    assert results[0]["metadata"]["page"] == 1

    print("\nRetrieved evidence:", results[0]["text"])
    print("Source metadata:", results[0]["metadata"])
    print("Cosine distance:", results[0]["distance"])