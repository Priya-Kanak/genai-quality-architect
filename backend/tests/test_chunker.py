import pytest

from app.rag.chunker import chunk_documents

def test_chunk_document():
    """Verify documents are split into overlapping chunks.
    """
    
    pages = [
        {
           "text" : (
               "one two three four five six "
                "seven eight nine ten eleven twelve"
                ),
           "metadata" : {
               "source":"test.pdf",
               "page": 1,
           }, 
        }
    ]
    
    chunks = chunk_documents(
        pages,
        chunk_size=5,
        chunk_overlap=2,
    )
    
    assert len(chunks) > 1
    
    assert chunks[0]["text"] == "one two three four five"
    assert chunks[1]["text"] == "four five six seven eight"
    
    assert chunks[0]["metadata"]["source"] == "test.pdf"
    assert chunks[0]["metadata"]["page"] == 1
    
    assert( 
           chunks[0]["metadata"]["chunk_id"] == "test.pdf_p1_c1"
           )
    
def test_invalid_chunk_size():
    """Verify zero chunk size is rejected."""
    
    with pytest.raises(ValueError):
        chunk_documents(
            [],
            chunk_size=0,
            chunk_overlap=0,
        )
        
        
def test_overlap_cannot_equal_chunk_size():
    """Verify overlap must be smaller than chunk size."""
    
    with pytest.raises(ValueError):
        chunk_documents(
                    [],
                    chunk_size=100,
                    chunk_overlap=100,
                )