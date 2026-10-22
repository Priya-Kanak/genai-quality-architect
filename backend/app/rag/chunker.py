def chunk_documents(
    pages: list[dict],
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> list[dict]:
    
    """Split document pages into overlapping text chunks."""
    
    if chunk_size <=0:
        raise ValueError("chunk_size must be greater than 0")
    
    if chunk_overlap <0 :
        raise ValueError("chunk_overlap cannot be negative")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    chunks = []
    
    for page in pages:
        words = page["text"].split()
        metadata = page["metadata"]
        
        start = 0
        chunk_number = 1
        step = chunk_size - chunk_overlap
        
        while start < len(words):
            end = start + chunk_size
            
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata":{
                             **metadata,
                            "chunk_id": (
                                f"{metadata['source']}"
                                f"_p{metadata['page']}"
                                f"_c{chunk_number}"
                            ),
                            "chunk_number": chunk_number, 
                        },
                    }
                )
            
            if end >= len(words):
                break
            
            start += step
            chunk_number += 1
            
            
    return chunks
        