from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
import tempfile
import os
import zipfile

from trocr import TrOCRPdfExtractor, answer_with_rag
from pipeline import extract_machine_images

app = FastAPI(title="PDF Processing Pipeline API", version="1.0.0")
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024


def _remove_file_safely(file_path: str) -> None:
    try:
        if file_path and os.path.isfile(file_path):
            os.remove(file_path)
    except OSError:
        pass


def _create_response_zip_path() -> str:
    fd, zip_path = tempfile.mkstemp(prefix="machine_crops_", suffix=".zip")
    os.close(fd)
    return zip_path


def _count_files_in_dir(folder_path: str) -> int:
    if not os.path.isdir(folder_path):
        return 0
    return sum(1 for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry)))


def _build_filter_stats(filter_root: str) -> dict[str, int]:
    animated_count = int(_count_files_in_dir(os.path.join(filter_root, "animated")))
    real_count = int(_count_files_in_dir(os.path.join(filter_root, "real")))
    return {
        "animated": animated_count,
        "real": real_count,
        "total": animated_count + real_count,
    }


def _is_likely_pdf(content_type: str | None, filename: str | None) -> bool:
    if content_type == "application/pdf":
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


def _validate_pdf_signature(pdf_path: str) -> None:
    with open(pdf_path, "rb") as pdf_file:
        header = pdf_file.read(5)
    if header != b"%PDF-":
        raise HTTPException(status_code=400, detail="Input is not a valid PDF.")


async def _save_uploaded_pdf(file: UploadFile, destination_path: str) -> None:
    if not _is_likely_pdf(file.content_type, file.filename):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    total_size = 0
    with open(destination_path, "wb") as output_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_PDF_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="PDF too large. Max size is 50 MB.")
            output_file.write(chunk)

    _validate_pdf_signature(destination_path)


async def resolve_pdf_input(file: UploadFile | None, pdf_url: str | None, destination_path: str) -> None:
    normalized_pdf_url = (pdf_url or "").strip().lower()
    if normalized_pdf_url and normalized_pdf_url not in {"null", "none", "undefined", "string"}:
        raise HTTPException(status_code=400, detail="pdf_url is no longer supported. Please upload a PDF file.")

    if file is None:
        raise HTTPException(status_code=400, detail="Provide a 'file' PDF upload.")

    await _save_uploaded_pdf(file, destination_path)


async def extract_text_from_upload(
    file: UploadFile | None,
    pdf_url: str | None,
    use_tesseract: bool,
    dpi: int,
) -> str:
    extractor = TrOCRPdfExtractor(use_tesseract=use_tesseract)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        await resolve_pdf_input(file=file, pdf_url=pdf_url, destination_path=pdf_path)

        images = extractor.pdf_to_images(pdf_path, dpi=dpi)
        extracted_text = []
        for page_num, image in enumerate(images, start=1):
            text = extractor.ocr_image(image)
            extracted_text.append(f"\n--- Page {page_num} ---\n{text}")

    return "\n".join(extracted_text)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "PDF Processing Pipeline API"}


@app.post("/ocr")
async def ocr_pdf(
    file: UploadFile | None = File(None),
    pdf_url: str | None = Form(None),
    use_tesseract: bool = Form(True),
    dpi: int = Form(300),
    include_text: bool = Form(False),
    question: str = Form(
        "Summarize the document with the main purpose, key technical details, and any important caveats."
    ),
    top_k: int = Form(4),
):
    """Extract text from PDF using OCR and answer a question with RAG"""
    try:
        full_text = await extract_text_from_upload(file, pdf_url, use_tesseract, dpi)
        rag_result = answer_with_rag(full_text, question=question, top_k=top_k)

        if include_text:
            return {
                "question": question,
                "summary": rag_result["answer"],
                "answer": rag_result["answer"],
                "retrieved_chunks": rag_result["retrieved_chunks"],
                "text": full_text,
            }

        return {
            "question": question,
            "summary": rag_result["answer"],
            "answer": rag_result["answer"],
            "retrieved_chunks": rag_result["retrieved_chunks"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
async def ask_question(
    file: UploadFile | None = File(None),
    pdf_url: str | None = Form(None),
    question: str = Form(...),
    use_tesseract: bool = Form(True),
    dpi: int = Form(300),
    include_text: bool = Form(False),
    top_k: int = Form(4),
):
    """Ask a question about an uploaded PDF using OCR + RAG retrieval."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="'question' cannot be empty.")

    try:
        full_text = await extract_text_from_upload(file, pdf_url, use_tesseract, dpi)
        rag_result = answer_with_rag(full_text, question=question, top_k=top_k)

        response = {
            "question": question,
            "answer": rag_result["answer"],
            "retrieved_chunks": rag_result["retrieved_chunks"],
        }
        if include_text:
            response["text"] = full_text

        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/process")
async def process_pipeline(
    file: UploadFile | None = File(None),
    pdf_url: str | None = Form(None),
    use_tesseract: bool = Form(True),
    dpi: int = Form(300),
    question: str = Form(
        "Summarize the document with the main purpose, key technical details, and any important caveats."
    ),
    top_k: int = Form(4),
    conf_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    filter_crops: bool = Form(True),
    return_crops: bool = Form(False),
):
    """Full pipeline: OCR + document Q&A + machine image extraction"""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pdf_path = os.path.join(tmpdir, "input.pdf")
            await resolve_pdf_input(file=file, pdf_url=pdf_url, destination_path=pdf_path)

            # Define output paths
            text_output = os.path.join(tmpdir, "output.txt")
            summary_output = os.path.join(tmpdir, "summary.txt")
            crops_folder = os.path.join(tmpdir, "machine_crops")
            filter_output_root = os.path.join(tmpdir, "filtered_crops")

            # Step 1: OCR & Summarization
            extractor = TrOCRPdfExtractor(use_tesseract=use_tesseract)
            full_text = extractor.extract_text_from_pdf(pdf_path)

            with open(text_output, "w", encoding="utf-8") as f:
                f.write(full_text)

            rag_result = answer_with_rag(
                full_text,
                question=question,
                top_k=top_k,
            )
            summary = rag_result["answer"]
            with open(summary_output, "w", encoding="utf-8") as f:
                f.write(summary)

            # Step 2: Machine Image Extraction
            crop_count = extract_machine_images(
                pdf_path=pdf_path,
                output_folder=crops_folder,
                dpi=dpi,
                conf_threshold=conf_threshold,
                text_threshold=text_threshold,
                apply_crop_filter=filter_crops,
                filter_output_root=filter_output_root,
            )

            filter_stats = _build_filter_stats(filter_output_root) if filter_crops else {"animated": 0, "real": 0, "total": 0}

            # Build response
            response_data = {
                "status": "success",
                "text": full_text,
                "question": question,
                "summary": summary,
                "answer": summary,
                "retrieved_chunks": rag_result["retrieved_chunks"],
                "machine_crops_count": int(crop_count),
                "filtered_crops": filter_stats,
            }

            # If requested, return crops as zip file
            if return_crops and crop_count > 0:
                zip_path = _create_response_zip_path()
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for img_file in os.listdir(crops_folder):
                        img_path = os.path.join(crops_folder, img_file)
                        if os.path.isfile(img_path):
                            zf.write(img_path, arcname=f"machine_crops/{img_file}")

                    if filter_crops:
                        for class_name in ("animated", "real"):
                            class_dir = os.path.join(filter_output_root, class_name)
                            if not os.path.isdir(class_dir):
                                continue
                            for img_file in os.listdir(class_dir):
                                img_path = os.path.join(class_dir, img_file)
                                if os.path.isfile(img_path):
                                    zf.write(img_path, arcname=f"filtered_crops/{class_name}/{img_file}")
                
                return FileResponse(
                    path=zip_path,
                    media_type="application/zip",
                    filename="machine_crops.zip",
                    background=BackgroundTask(_remove_file_safely, zip_path),
                )

            return JSONResponse(status_code=200, content=response_data)

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}") from exc


@app.post("/extract-images")
async def extract_images_only(
    file: UploadFile | None = File(None),
    pdf_url: str | None = Form(None),
    dpi: int = Form(300),
    conf_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    filter_crops: bool = Form(True),
):
    """Extract machine images from PDF only (no OCR/summary)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            pdf_path = os.path.join(tmpdir, "input.pdf")
            await resolve_pdf_input(file=file, pdf_url=pdf_url, destination_path=pdf_path)

            crops_folder = os.path.join(tmpdir, "machine_crops")
            filter_output_root = os.path.join(tmpdir, "filtered_crops")

            crop_count = extract_machine_images(
                pdf_path=pdf_path,
                output_folder=crops_folder,
                dpi=dpi,
                conf_threshold=conf_threshold,
                text_threshold=text_threshold,
                apply_crop_filter=filter_crops,
                filter_output_root=filter_output_root,
            )

            filter_stats = _build_filter_stats(filter_output_root) if filter_crops else {"animated": 0, "real": 0, "total": 0}

            if crop_count == 0:
                return {
                    "status": "success",
                    "machine_crops_count": 0,
                    "filtered_crops": filter_stats,
                    "message": "No machine objects detected",
                }

            # Return crops as zip file
            zip_path = _create_response_zip_path()
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_file in os.listdir(crops_folder):
                    img_path = os.path.join(crops_folder, img_file)
                    if os.path.isfile(img_path):
                        zf.write(img_path, arcname=f"machine_crops/{img_file}")

                if filter_crops:
                    for class_name in ("animated", "real"):
                        class_dir = os.path.join(filter_output_root, class_name)
                        if not os.path.isdir(class_dir):
                            continue
                        for img_file in os.listdir(class_dir):
                            img_path = os.path.join(class_dir, img_file)
                            if os.path.isfile(img_path):
                                zf.write(img_path, arcname=f"filtered_crops/{class_name}/{img_file}")

            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="machine_crops.zip",
                background=BackgroundTask(_remove_file_safely, zip_path),
            )

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Image extraction error: {str(exc)}") from exc
