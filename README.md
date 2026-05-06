---
title: RAG Summarizer
description: A Retrieval-Augmented Generation (RAG) document QA and summarization tool built around OCR and retrieval.
---

# RAG Summarizer

A lightweight pipeline that extracts text from documents/images (OCR), chunks and indexes the text, then performs retrieval-augmented question answering (RAG) and summarization.

## Features
- OCR extraction from PDFs and images
- Chunking and retrieval of relevant passages for context
- API endpoints for OCR, document Q&A, and processing
- CLI for quick local summarization and QA

## Requirements
- Python 3.10+
- See `requirements.txt` for full dependency list

## Quick start
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the built-in CLI to process a PDF:

```powershell
python pipeline.py --pdf input.pdf --question "What are the key findings?"
```

3. Or run the API (if available in this project): start the app and use the endpoints below.

## API Endpoints
- `POST /ocr` — upload a PDF/image and optionally include `question`. Returns OCR text and (when asked) an answer plus retrieved chunks.
- `POST /ask` — submit a `question` for document Q&A (requires previously-processed document context).
- `POST /process` — performs OCR + image extraction; accepts `question` and `top_k` to return an answer for the input document.

Example (curl):

```bash
curl -X POST "http://localhost:8000/ocr" -F "file=@input.pdf" -F "question=What are the key findings?"
```

## Project layout
- `pipeline.py` — main processing pipeline and CLI
- `api.py` / `main.py` — API server entrypoints (if present)
- `trocr.py`, `yolo.py` — helper scripts for OCR/imaging
- `dataset/`, `images/` — sample inputs and resources

## License
Specify your license here (e.g. MIT).

---
If you want, I can also add a CONTRIBUTING or LICENSE file and push changes to the remote repository.
