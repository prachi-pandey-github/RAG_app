import argparse
import os

import cv2
import numpy as np
import torch
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from test import filter_machine_crops
from trocr import TrOCRPdfExtractor, answer_with_rag


TEXT_PROMPT = (
    "machine, computer, battery, "
    "mechanical equipment, construction equipment, "
    "factory machine, computer motherboard, "
    "circuit board, PCB, electronic hardware, device"
)


def resolve_poppler_path() -> str | None:
    env_poppler = os.environ.get("POPPLER_PATH")
    if env_poppler and os.path.isdir(env_poppler):
        return env_poppler

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "poppler-25.12.0", "Library", "bin"),
        os.path.join(os.getcwd(), "poppler-25.12.0", "Library", "bin"),
    ]

    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "pdfinfo.exe")):
            return candidate

    return None


def _render_pdf_with_pymupdf(pdf_path: str, dpi: int) -> list[Image.Image]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Poppler is not available and PyMuPDF is not installed. "
            "Install pymupdf or provide Poppler executables so PDFs can be rendered to images."
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


def extract_machine_images(
    pdf_path: str,
    output_folder: str,
    dpi: int,
    conf_threshold: float,
    text_threshold: float,
    apply_crop_filter: bool = True,
    filter_output_root: str = "dataset",
) -> int:
    os.makedirs(output_folder, exist_ok=True)

    print("\nLoading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Converting PDF to images for object detection...")
    poppler_path = resolve_poppler_path()

    try:
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    except PDFInfoNotInstalledError as exc:
        print("Poppler not found, falling back to PyMuPDF rendering...")
        pages = _render_pdf_with_pymupdf(pdf_path, dpi=dpi)

    crop_count = 0

    for page_index, page in enumerate(pages, start=1):
        print(f"Processing page {page_index} for image extraction...")

        image = page.convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=conf_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        except TypeError as exc:
            if "box_threshold" not in str(exc):
                raise
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=conf_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]

        for box, score in zip(boxes, scores):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = box.int().tolist()
            crop = image_cv[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            save_path = os.path.join(output_folder, f"page{page_index}_crop{crop_count}.jpg")
            cv2.imwrite(save_path, crop)
            crop_count += 1

    if apply_crop_filter and crop_count > 0:
        print("Applying animated/real crop filter...")
        try:
            filtered = filter_machine_crops(
                crops_folder=output_folder,
                output_root=filter_output_root,
                move_files=False,
            )
            print(
                "Filtered crops:",
                f"animated={filtered.get('animated', 0)},",
                f"real={filtered.get('real', 0)},",
                f"skipped={filtered.get('skipped', 0)}",
            )
        except Exception as exc:
            print(f"Warning: crop filtering failed: {exc}")

    return crop_count


def run_pipeline(
    pdf_path: str,
    use_tesseract: bool,
    dpi: int,
    output_text_path: str,
    summary_path: str,
    question: str,
    top_k: int,
    crops_folder: str,
    conf_threshold: float,
    text_threshold: float,
    apply_crop_filter: bool,
    filter_output_root: str,
) -> None:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Starting OCR + RAG answer generation...")
    extractor = TrOCRPdfExtractor(use_tesseract=use_tesseract)
    full_text = extractor.extract_text_from_pdf(pdf_path)

    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(full_text)
    print(f"OCR text saved to: {output_text_path}")

    rag_result = answer_with_rag(full_text, question=question, top_k=top_k)
    summary = rag_result["answer"]
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)
    print(f"Summary saved to: {summary_path}")
    print(f"Retrieved chunks used: {len(rag_result['retrieved_chunks'])}")

    print("\nStarting machine image extraction...")
    crop_count = extract_machine_images(
        pdf_path=pdf_path,
        output_folder=crops_folder,
        dpi=dpi,
        conf_threshold=conf_threshold,
        text_threshold=text_threshold,
        apply_crop_filter=apply_crop_filter,
        filter_output_root=filter_output_root,
    )

    print("\n=================================")
    print("Pipeline complete ✅")
    print(f"Text output: {output_text_path}")
    print(f"Summary output: {summary_path}")
    print(f"Image crops folder: {crops_folder}")
    print(f"Total machine objects saved: {crop_count}")
    print("=================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR + RAG answer generation + machine image extraction pipeline")
    parser.add_argument("--pdf", default="input.pdf", help="Path to input PDF")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF page rendering")
    parser.add_argument("--use-tesseract", action="store_true", default=True, help="Use Tesseract OCR (default)")
    parser.add_argument("--use-trocr", action="store_true", help="Use TrOCR instead of Tesseract")
    parser.add_argument("--output-text", default="output.txt", help="Path for extracted OCR text")
    parser.add_argument("--output-summary", default="summary.txt", help="Path for summary output")
    parser.add_argument(
        "--question",
        default="Summarize the document with the main purpose, key technical details, and any important caveats.",
        help="Question to ask against the extracted document text",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of document chunks to retrieve for the answer")
    parser.add_argument("--crops-folder", default="machine_crops", help="Folder for extracted image crops")
    parser.add_argument("--conf-threshold", type=float, default=0.35, help="Object detection confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Grounding text threshold")
    parser.add_argument(
        "--filter-crops",
        action="store_true",
        default=True,
        help="Filter extracted crops into animated/real folders (default: enabled)",
    )
    parser.add_argument(
        "--no-filter-crops",
        action="store_false",
        dest="filter_crops",
        help="Disable animated/real filtering for extracted crops",
    )
    parser.add_argument(
        "--filter-output-root",
        default="dataset",
        help="Output root for filtered crops (contains animated/ and real/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_tesseract = True
    if args.use_trocr:
        use_tesseract = False

    run_pipeline(
        pdf_path=args.pdf,
        use_tesseract=use_tesseract,
        dpi=args.dpi,
        output_text_path=args.output_text,
        summary_path=args.output_summary,
        question=args.question,
        top_k=args.top_k,
        crops_folder=args.crops_folder,
        conf_threshold=args.conf_threshold,
        text_threshold=args.text_threshold,
        apply_crop_filter=args.filter_crops,
        filter_output_root=args.filter_output_root,
    )
