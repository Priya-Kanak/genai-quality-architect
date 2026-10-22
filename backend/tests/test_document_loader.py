import pytest

from app.rag.document_loader import load_pdf

def test_document_not_found():
    """Verify missing documents fail with a clea error."""
    
    with pytest.raises(FileNotFoundError):
        load_pdf(
            "data/documents/document_that_does_not_exist.pdf"
        )
        
def test_load_real_pdf():
    """Verify text and metadata can be extracted from a real PDF."""
    
    pages = load_pdf(
        "data/documents/policy_document.pdf"
                     )
    assert len(pages) > 0
    
    first_page = pages[0]
    
    assert "text" in first_page
    assert "metadata" in first_page
    
    assert len(first_page["text"])>0
    
    assert first_page["metadata"]["source"] == "policy_document.pdf"
    assert first_page["metadata"]["page"] == 1