# PDF Processing Pipeline API Guide

## Start the Server

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs` (Swagger UI)
Alternative docs: `http://localhost:8000/redoc` (ReDoc)

---

## Available Endpoints

### 1. Health Check
```
GET /health
```
Returns the server status.

**Response:**
```json
{
  "status": "ok",
  "service": "PDF Processing Pipeline API"
}
```

---

### 2. OCR Only
```
POST /ocr
```
Extract text and generate summary from PDF (no image extraction).

**Parameters:**
- `file` (file, optional*): PDF file to process
- `pdf_url` (string, optional*): Public HTTP/HTTPS URL of a PDF
- `use_tesseract` (boolean, default: `true`): Use Tesseract OCR instead of TrOCR
- `dpi` (integer, default: `300`): DPI for PDF rendering
- `include_text` (boolean, default: `false`): Include full extracted text in response

\*Provide exactly one of `file` or `pdf_url`.

**Response:**
```json
{
  "summary": "Summary of the document...",
  "text": "Full extracted text..." // Only if include_text=true
}
```

**Example with curl:**
```bash
# Using file upload
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@input.pdf" \
  -F "use_tesseract=true" \
  -F "dpi=300" \
  -F "include_text=true"

# Using PDF URL
curl -X POST "http://localhost:8000/ocr" \
  -F "pdf_url=https://example.com/sample.pdf" \
  -F "use_tesseract=true" \
  -F "dpi=300" \
  -F "include_text=true"
```

---

### 3. Full Pipeline (OCR + Summary + Image Extraction)
```
POST /process
```
Complete pipeline: OCR text extraction, summarization, and machine image detection.

**Parameters:**
- `file` (file, optional*): PDF file to process
- `pdf_url` (string, optional*): Public HTTP/HTTPS URL of a PDF
- `use_tesseract` (boolean, default: `true`): Use Tesseract OCR
- `dpi` (integer, default: `300`): DPI for PDF rendering
- `conf_threshold` (float, default: `0.35`): Object detection confidence threshold
- `text_threshold` (float, default: `0.25`): Grounding text threshold
- `return_crops` (boolean, default: `false`): Return detected machine images as ZIP file

\*Provide exactly one of `file` or `pdf_url`.

**Response (JSON):**
```json
{
  "status": "success",
  "text": "Extracted full text...",
  "summary": "Document summary...",
  "machine_crops_count": 15
}
```

**Response (when return_crops=true):**
- Returns a ZIP file containing all extracted machine images

**Example with curl:**
```bash
# Get JSON response with all data (file upload)
curl -X POST "http://localhost:8000/process" \
  -F "file=@input.pdf" \
  -F "use_tesseract=true" \
  -F "dpi=300" \
  -F "conf_threshold=0.35" \
  -F "text_threshold=0.25" \
  -F "return_crops=false"

# Get JSON response with all data (PDF URL)
curl -X POST "http://localhost:8000/process" \
  -F "pdf_url=https://example.com/sample.pdf" \
  -F "use_tesseract=true" \
  -F "dpi=300" \
  -F "conf_threshold=0.35" \
  -F "text_threshold=0.25" \
  -F "return_crops=false"

# Get ZIP file with detected machine images
curl -X POST "http://localhost:8000/process" \
  -F "file=@input.pdf" \
  -F "return_crops=true" \
  -o crops.zip
```

---

### 4. Extract Machine Images Only
```
POST /extract-images
```
Extract only machine/object images from PDF (no OCR/summarization).

**Parameters:**
- `file` (file, optional*): PDF file to process
- `pdf_url` (string, optional*): Public HTTP/HTTPS URL of a PDF
- `dpi` (integer, default: `300`): DPI for PDF rendering
- `conf_threshold` (float, default: `0.35`): Object detection confidence threshold
- `text_threshold` (float, default: `0.25`): Grounding text threshold

\*Provide exactly one of `file` or `pdf_url`.

**Response:**
- If crops found: Returns a ZIP file with extracted images
- If no crops found: Returns JSON with count and message

**Example with curl:**
```bash
# Using file upload
curl -X POST "http://localhost:8000/extract-images" \
  -F "file=@input.pdf" \
  -F "dpi=300" \
  -O -J

# Using PDF URL
curl -X POST "http://localhost:8000/extract-images" \
  -F "pdf_url=https://example.com/sample.pdf" \
  -F "dpi=300" \
  -O -J
```

---

## Model Details

### OCR
- **Default:** Tesseract OCR (local machine installation)
  - Path: `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows)
  - Falls back to PATH if not found
- **Alternative:** TrOCR (microsoft/trocr-base-printed)

### Summarization
- **Model:** Llama 3.3 70B (via Groq API)
- **Requires:** GROQ_API_KEY environment variable
  - Get key: https://console.groq.com

### Image Detection
- **Model:** Grounding DINO (IDEA-Research/grounding-dino-base)
- **Type:** Zero-shot object detection
- **Objects detected:** Machines, computers, batteries, equipment, PCBs, etc.

---

## Examples

### Python Client Example

```python
import requests

# File to process
pdf_file = "input.pdf"

# Full pipeline
with open(pdf_file, "rb") as f:
    response = requests.post(
        "http://localhost:8000/process",
        files={"file": f},
        data={
            "use_tesseract": True,
            "dpi": 300,
            "conf_threshold": 0.35,
            "text_threshold": 0.25,
            "return_crops": False
        }
    )
    result = response.json()
    print(f"Summary: {result['summary']}")
    print(f"Machine objects found: {result['machine_crops_count']}")

# Get machine images as ZIP
with open(pdf_file, "rb") as f:
    response = requests.post(
        "http://localhost:8000/process",
        files={"file": f},
        data={"return_crops": True}
    )
    with open("crops.zip", "wb") as zf:
        zf.write(response.content)
```

---

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (e.g., non-PDF file)
- `500` - Server error (e.g., model loading failure)

Error responses include detail message:
```json
{
  "detail": "Only PDF files are supported."
}
```

---

## Troubleshooting

1. **Tesseract not found:**
   - Install: `choco install tesseract` (Windows) or `brew install tesseract` (Mac)
   - Set environment: `TESSERACT_PATH=C:\Program Files\Tesseract-OCR`

2. **GROQ API key missing:**
   - Create `.env` file in project root
   - Add: `GROQ_API_KEY=your_api_key_here`

3. **CUDA/GPU issues:**
   - Grounding DINO will auto-fallback to CPU
   - Takes longer but works without GPU

---

## Performance Tips

- **DPI 300:** High quality, slower processing
- **DPI 150:** Balanced quality/speed
- **conf_threshold 0.35:** Default, balanced detection
- **conf_threshold 0.5+:** More conservative, fewer false positives
- Use `/extract-images` if you only need crops (faster)

