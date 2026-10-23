
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import FastAPI
from fastapi import File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.services.document_service import extract_pdf_text
from backend.app.services.chunking_service import chunk_document_pages
from backend.app.services.vector_store_service import (
    add_chunks_to_vector_index,
    search_vector_index,
)
from backend.app.services.rag_service import answer_question


app = FastAPI(
    title="Banking Controls Intelligence API",
    description="A document-grounded banking controls research platform.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:5174', 'http://127.0.0.1:5174'],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


MAX_FILE_SIZE = 10 * 1024 * 1024

UPLOAD_DIR = Path("data") / "uploads"
PROCESSED_DIR = Path("data") / "processed"
VECTOR_INDEX_DIR = Path("data") / "vector_index"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    document_id: str | None = Field(default=None, max_length=100)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    document_id: str | None = Field(default=None, max_length=100)


@app.get("/", tags=["System"])
def root():
    return {
        "name": "Banking Controls Intelligence API",
        "version": "0.3.0",
        "status": "running",
    }


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The PDF exceeds the 10 MB upload limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            prefix="upload_",
            dir=UPLOAD_DIR,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        extracted = extract_pdf_text(str(temp_path))

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="The PDF could not be read or saved.",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    if extracted["pages_with_text"] == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No selectable text or readable OCR text was found. "
                "The PDF may be blank or difficult to read."
            ),
        )

    document_id = str(uuid4())
    safe_filename = Path(filename).name

    chunks = chunk_document_pages(
        pages=extracted["pages"],
        document_id=document_id,
        filename=safe_filename,
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No searchable text chunks could be created from this PDF.",
        )

    chunk_records = [
        {
            "text": chunk.text,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    stored_document = {
        "document_id": document_id,
        "filename": safe_filename,
        "total_pages": extracted["total_pages"],
        "pages_with_text": extracted["pages_with_text"],
        "chunks": chunk_records,
    }

    output_path = PROCESSED_DIR / f"{document_id}.json"

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(
                stored_document,
                output_file,
                ensure_ascii=False,
                indent=2,
            )

        add_chunks_to_vector_index(
            chunk_records,
            VECTOR_INDEX_DIR,
        )

    except (OSError, ValueError, RuntimeError) as exc:
        output_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=(
                "The document could not be saved and indexed. "
                "Please retry."
            ),
        ) from exc

    return {
        "message": "PDF processed, chunked, and indexed successfully.",
        "document_id": document_id,
        "filename": safe_filename,
        "total_pages": extracted["total_pages"],
        "pages_with_text": extracted["pages_with_text"],
        "chunks_created": len(chunks),
        "vectors_indexed": len(chunks),
    }


@app.post("/api/v1/search", tags=["Search"])
def search(request: SearchRequest):
    try:
        results = search_vector_index(
            query=request.query,
            index_dir=VECTOR_INDEX_DIR,
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except FileNotFoundError:
        results = []

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The vector search could not be completed. "
                "Check the index and query."
            ),
        ) from exc

    return {
        "query": request.query,
        "document_id": request.document_id,
        "results_count": len(results),
        "results": results,
    }


@app.post("/api/v1/ask", tags=["RAG"])
def ask(request: AskRequest):
    try:
        result = answer_question(
            query=request.query,
            index_dir=VECTOR_INDEX_DIR,
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="No vector index exists yet. Upload a PDF first.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="The question could not be processed. Check the request.",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "query": request.query,
        **result,
    }
