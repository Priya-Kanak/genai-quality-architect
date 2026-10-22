from pathlib import Path

from pypdf import PdfReader

def load_pdf(file_path: str) -> list[dict]:
    """Load test and metadat from a PDF document."""
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )
        
    reader = PdfReader(path)
    
    pages = []
    
    for page_number, page in enumerate(reader.pages, start = 1):
        text = page.extract_text()
        
        if text and text.strip():
            pages.append(
                {
                    "text": text.strip(),
                    "metadata":{
                        "source": path.name,
                        "page": page_number,
                    },
                }
            )
    return pages       