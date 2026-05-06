# import os
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# from ultralytics import YOLO
# from PIL import Image

# # ===============================
# # CONFIG
# # ===============================
# PDF_PATH = "input.pdf"
# OUTPUT_FOLDER = "machinery_crops"
# YOLO_MODEL_PATH = "yolov8n.pt"  # Replace with your custom machinery model if available
# CONF_THRESHOLD = 0.25

# # Create output folder
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Load YOLO model
# model = YOLO(YOLO_MODEL_PATH)

# # Configure Poppler path for pdf2image
# poppler_candidate = os.path.join(os.getcwd(), "poppler-25.12.0", "Library", "bin")
# POPPLER_PATH = poppler_candidate if os.path.exists(poppler_candidate) else None

# # ===============================
# # STEP 1: Convert PDF to Images
# # ===============================
# print("Converting PDF to images...")
# pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)

# print(f"Total pages found: {len(pages)}")

# crop_count = 0

# # ===============================
# # STEP 2: Process Each Page
# # ===============================
# for page_index, page in enumerate(pages):
#     print(f"\nProcessing Page {page_index + 1}...")
    
#     # Convert PIL image to OpenCV format
#     page_np = np.array(page)
#     page_cv = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
    
#     # Run YOLO detection
#     results = model(page_cv)
    
#     for result in results:
#         boxes = result.boxes
        
#         for box in boxes:
#             confidence = float(box.conf[0])
#             class_id = int(box.cls[0])
#             class_name = model.names[class_id]
            
#             print(f"  Detected: {class_name} (confidence: {confidence:.2f})")

#             # Save all detected objects above confidence threshold
#             if confidence > CONF_THRESHOLD:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
                
#                 # Add padding around the bounding box for better crops
#                 padding = 20  # pixels
#                 img_height, img_width = page_cv.shape[:2]
                
#                 x1 = max(0, x1 - padding)
#                 y1 = max(0, y1 - padding)
#                 x2 = min(img_width, x2 + padding)
#                 y2 = min(img_height, y2 + padding)
                
#                 crop = page_cv[y1:y2, x1:x2]
                
#                 crop_filename = os.path.join(
#                     OUTPUT_FOLDER,
#                     f"page{page_index+1}_crop{crop_count}.jpg"
#                 )
                
#                 cv2.imwrite(crop_filename, crop)
#                 print(f"  Saved: {crop_filename}")
#                 crop_count += 1

# print(f"\nDone ✅ Total machinery objects saved: {crop_count}")


import os
import cv2
import torch
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# =====================================
# CONFIG
# =====================================
PDF_PATH = "input.pdf"
OUTPUT_FOLDER = "machine_crops"
CONF_THRESHOLD = 0.35

# Broad machine detection prompt
TEXT_PROMPT = (
    "machine, computer, battery, "
    "mechanical equipment, construction equipment, "
    "factory machine, computer motherboard, "
    "circuit board, PCB, electronic hardware, device"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def resolve_poppler_path():
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

# =====================================
# LOAD GROUNDING DINO
# =====================================
print("Loading Grounding DINO model...")

model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =====================================
# CONVERT PDF TO IMAGES
# =====================================
print("Converting PDF to images...")
poppler_path = resolve_poppler_path()

try:
    pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=poppler_path)
except PDFInfoNotInstalledError as exc:
    raise RuntimeError(
        "Poppler not found. Set POPPLER_PATH or ensure 'poppler-25.12.0/Library/bin' exists next to this script."
    ) from exc

print(f"Total pages: {len(pages)}")

crop_count = 0

# =====================================
# PROCESS EACH PAGE
# =====================================
for page_index, page in enumerate(pages):

    print(f"\nProcessing Page {page_index+1}")

    image = page.convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=CONF_THRESHOLD,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
    except TypeError as exc:
        if "box_threshold" not in str(exc):
            raise
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=CONF_THRESHOLD,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]

    print(f"Detections found: {len(boxes)}")

    for box, score, label in zip(boxes, scores, labels):

        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.int().tolist()

        crop = image_cv[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        save_path = os.path.join(
            OUTPUT_FOLDER,
            f"page{page_index+1}_crop{crop_count}.jpg"
        )

        cv2.imwrite(save_path, crop)
        crop_count += 1

print("\n=================================")
print(f"Done ✅ Total machine objects saved: {crop_count}")
print("=================================")