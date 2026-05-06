from dataclasses import dataclass
import os
import re
import warnings

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract

# Load environment variables from .env file
load_dotenv()

# Suppress PIL decompression bomb warning
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)


SUMMARY_QUESTION = (
    "Summarize the document with the main purpose, key technical details, and any important caveats."
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float


def chunk_document_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if not cleaned_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            split_at = cleaned_text.rfind(". ", start, end)
            if split_at == -1 or split_at <= start + max(1, chunk_size // 3):
                split_at = cleaned_text.rfind(" ", start, end)
            if split_at == -1 or split_at <= start:
                split_at = end
            else:
                split_at += 1
        else:
            split_at = end

        chunk = cleaned_text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= text_length:
            break

        start = max(split_at - overlap, start + 1)

    return chunks


def _render_pdf_with_pymupdf(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Poppler is not available and PyMuPDF is not installed. "
            "Install pymupdf or provide Poppler so PDFs can be rendered to images."
        ) from exc

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pixmap.n < 4 else "RGBA"
            image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
            images.append(image.convert("RGB"))

    return images


class DocumentRAG:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks) if chunks else None

    @classmethod
    def from_text(cls, text: str, chunk_size: int = 1200, overlap: int = 200) -> "DocumentRAG":
        return cls(chunk_document_text(text, chunk_size=chunk_size, overlap=overlap))

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not self.chunks or self.matrix is None:
            return []

        query = question.strip()
        if not query:
            query = SUMMARY_QUESTION

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1][: max(1, top_k)]
        return [
            RetrievedChunk(chunk_id=index + 1, text=self.chunks[index], score=float(scores[index]))
            for index in ranked_indices
            if scores[index] > 0
        ]

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context was retrieved from the document."

        return "\n\n".join(
            f"[Chunk {chunk.chunk_id} | score={chunk.score:.3f}]\n{chunk.text}" for chunk in chunks
        )


def _generate_answer_from_context(question: str, context: str, api_key: str | None = None) -> str:
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found.\n"
                "Please:\n"
                "1. Create a Gemini API key in Google AI Studio\n"
                "2. Create a .env file with: GEMINI_API_KEY=your_key_here\n"
                "   OR set environment variable: $env:GEMINI_API_KEY='your_key_here'"
            )

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        prompt = (
            "You are a helpful RAG assistant. Answer only from the provided document context. "
            "If the context does not contain enough information, say so clearly. "
            "Be concise, factual, and cite the chunk numbers you used.\n\n"
            f"Question: {question}\n\n"
            f"Document context:\n{context}\n\n"
            "Return a direct answer followed by a short 'Sources:' line."
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        answer_text = (response.text or "").strip()
        if answer_text:
            return answer_text

        return _build_local_fallback_answer(question, context, "Gemini returned an empty response.")
    except Exception as exc:
        return _build_local_fallback_answer(question, context, str(exc))


def _build_local_fallback_answer(question: str, context: str, error_message: str) -> str:
    retrieved_blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
    if not retrieved_blocks:
        return (
            "I could not generate an LLM answer because the external model request failed, and no relevant "
            "document context was retrieved."
        )

    summary_lines: list[str] = []
    for block in retrieved_blocks[:3]:
        lines = block.splitlines()
        chunk_label = lines[0].strip() if lines else "[Chunk]"
        chunk_text = " ".join(lines[1:]).strip() if len(lines) > 1 else block
        sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        excerpt = sentences[0].strip() if sentences and sentences[0].strip() else chunk_text[:240].strip()
        summary_lines.append(f"- {chunk_label}: {excerpt}")

    lower_question = question.lower()
    if any(keyword in lower_question for keyword in ("summary", "summarize", "summarise")):
        intro = "Key points from the retrieved document context:"
    else:
        intro = "Answer based on the retrieved document context:"

    sources = ", ".join(block.splitlines()[0].strip() for block in retrieved_blocks[:3] if block.splitlines())
    return (
        f"{intro}\n"
        + "\n".join(summary_lines)
        + f"\n\nSources: {sources}\n"
        + f"Fallback used because the Gemini request failed: {error_message}"
    )


def answer_with_rag(
    text: str,
    question: str,
    api_key: str | None = None,
    top_k: int = 4,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> dict[str, object]:
    rag = DocumentRAG.from_text(text, chunk_size=chunk_size, overlap=overlap)
    retrieved_chunks = rag.retrieve(question, top_k=top_k)
    context = rag.format_context(retrieved_chunks)
    answer = _generate_answer_from_context(question, context, api_key=api_key)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": [
            {
                "chunk_id": int(chunk.chunk_id),
                "score": round(chunk.score, 4),
                "text": chunk.text,
            }
            for chunk in retrieved_chunks
        ],
    }


class TrOCRPdfExtractor:
    def __init__(self, use_tesseract=True, model_name="microsoft/trocr-base-printed", device=None):
        self.use_tesseract = use_tesseract
        
        if not use_tesseract:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
        else:
            # Set Tesseract path for Windows (common installation path)
            if os.name == "nt":
                tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                if os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Use bundled Poppler on Windows if present; otherwise let pdf2image find it in PATH.
        poppler_candidate = os.path.join(os.getcwd(), "poppler-25.12.0", "Library", "bin")
        self.poppler_path = poppler_candidate if os.path.exists(poppler_candidate) else None

    def pdf_to_images(self, pdf_path, dpi=300):
        try:
            return convert_from_path(pdf_path, dpi=dpi, poppler_path=self.poppler_path)
        except PDFInfoNotInstalledError:
            return _render_pdf_with_pymupdf(pdf_path, dpi=dpi)

    def ocr_image(self, image: Image.Image):
        if self.use_tesseract:
            # Use Tesseract for full-page OCR (better for printed documents)
            text = pytesseract.image_to_string(image)
            return text.strip()
        else:
            # Use TrOCR (better for handwritten text or line-by-line)
            import torch
            
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        images = self.pdf_to_images(pdf_path)

        extracted_text = []
        for page_num, image in enumerate(images, start=1):
            print(f"Processing page {page_num}...")
            text = self.ocr_image(image)
            extracted_text.append(f"\n--- Page {page_num} ---\n{text}")

        return "\n".join(extracted_text)


def summarize_with_gemini(text, api_key=None):
    """
    Summarize the extracted text using Gemini-powered retrieval-augmented generation.
    """
    print("Generating summary with Gemini retrieval-augmented generation...")
    result = answer_with_rag(text, SUMMARY_QUESTION, api_key=api_key)
    return result["answer"]


def summarize_with_llama(text, api_key=None):
    """Backward-compatible alias for legacy callers."""
    return summarize_with_gemini(text, api_key=api_key)


if __name__ == "__main__":
    pdf_path = "input.pdf"

    # Use Tesseract for full-page printed text extraction
    extractor = TrOCRPdfExtractor(use_tesseract=True)
    
    # Alternative: Use TrOCR (uncomment below for handwritten text)
    # extractor = TrOCRPdfExtractor(use_tesseract=False, model_name="microsoft/trocr-base-printed")

    text = extractor.extract_text_from_pdf(pdf_path)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("OCR completed. Text saved to output.txt")
    
    # Generate summary using Gemini-powered RAG
    try:
        summary = summarize_with_gemini(text)
        
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        
        print("\n" + "="*50)
        print("SUMMARY:")
        print("="*50)
        print(summary)
        print("="*50)
        print("\nSummary saved to summary.txt")
        
    except Exception as e:
        print(f"\nError generating summary: {e}")
        print("OCR text is still available in output.txt")
