import io
import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming 'main.py' contains your FastAPI app

client = TestClient(app)



def test_health_check():
    """Test the health check endpoint to ensure the app is running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Service running"}


def test_chat():
    """Test /chat endpoint."""
    
    # Send a test request to the /chat endpoint
    response = client.post("/chat", json={"user_query": "What is Python?"})
    
    # Assert the response is correct
    assert response.status_code == 200
    assert response.json() == {"answer": "Page: Python (programming language)\nSummary: Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.\nPython is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a \"batteries included\" language due to its comprehensive standard library.\nGuido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions. Python 2.7.18, released in 2020, was the last release of Python 2.\nPython consistently ranks as one of the most popular programming languages, and has gained widespread use in the machine learning community.\n\n"}


def test_upload_pdf_file():
    """Test uploading a valid PDF file."""
    
    # Prepare a simple PDF file content (you can generate a PDF or mock a simple one)
    pdf_content = io.BytesIO(b"%PDF-1.4 ...")  # Mock PDF header, can be simplified for testing purposes
    pdf_content.name = "sample.pdf"
    
    # Send the PDF file in the request
    response = client.post(
        "/upload",  # Assuming this is the endpoint for file upload
        files={"file": ("sample.pdf", pdf_content, "application/pdf")},
    )
    
    # Assert the response
    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.pdf",
        "message": "File uploaded successfully",  
    }