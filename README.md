BANKING CONTROLS INTELLIGENCE

ABOUT THE PROJECT

I built this project to make it easier to find information in lengthy banking documents. Instead of reading every page manually, users can upload a PDF and ask questions about its contents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and pass it to a language model before generating an answer.

FEATURES

- Upload PDF documents
- Extract text and split it into smaller chunks
- Generate text embeddings using Sentence Transformers
- Store and search vectors using FAISS
- Retrieve relevant document sections using semantic search
- Generate answers using a local language model through Ollama
- Return evidence and citation metadata through the API
- Handle questions when relevant evidence is unavailable
- Provide a web interface built with React and TypeScript

TECHNOLOGIES USED

Python
FastAPI
Sentence Transformers
FAISS
Ollama
Qwen 2.5 3B
React
TypeScript
Vite
pytest

HOW IT WORKS

1. The user uploads a PDF document.
2. The backend extracts the text and divides it into overlapping chunks.
3. Sentence Transformers converts the chunks into embeddings.
4. FAISS stores the vectors and retrieves relevant chunks for a question.
5. The retrieved text is passed to the language model as context.
6. The API returns the answer along with evidence and citation metadata.

PROJECT STRUCTURE

backend - API, document processing, embeddings, retrieval, RAG services and tests
frontend - React and TypeScript user interface
requirements.txt - Python dependencies

HOW TO RUN THE PROJECT

Prerequisites

- Python 3.13
- Node.js and npm
- Git
- Ollama

1. Clone the repository

git clone https://github.com/kanika7098/Banking-Controls-Intelligence.git

cd Banking-Controls-Intelligence

2. Set up the Python environment

On Windows PowerShell, run:

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

3. Download the language model

Make sure Ollama is installed, then run:

ollama pull qwen2.5:3b

Ollama must be running before you send questions to the backend.

4. Start the backend

From the project root, run:

python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

API documentation is available at:

http://127.0.0.1:8000/docs

5. Start the frontend

Open a second terminal and run:

cd frontend
npm install
npm run dev

Open the local URL displayed by Vite.

RUNNING TESTS

From the project root, run:

python -m pytest backend/tests -v

The backend test suite passed all 41 tests in my latest recorded run.

CURRENT LIMITATIONS

Answer quality depends on the contents of the uploaded documents and the relevance of the retrieved evidence.

Citation behavior should be checked against real documents before relying on answers.

This project is a local prototype and is not a production banking or compliance system.

Do not upload confidential banking documents, credentials or other sensitive information to a public repository.

WHY I BUILT THIS

I wanted to build a practical project combining document processing, semantic search, embeddings, vector databases, APIs and large language models. Banking documents provide a useful context for exploring how these technologies can help people find information more efficiently.

FUTURE IMPROVEMENTS

Improve retrieval quality and citation validation.

Add more evaluation cases for answer relevance and evidence coverage.

Introduce document-level access controls and stronger handling of sensitive data.

Make deployment and setup easier.
