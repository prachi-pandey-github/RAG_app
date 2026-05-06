import os
import pickle
import shutil
from dataclasses import dataclass


DEFAULT_CLASSES = ["animated", "real"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CropFilter:
    clip_model: object
    preprocess: object
    classifier: object
    device: str
    classes: list[str]


def load_crop_filter(classifier_path: str = "classifier.pkl") -> CropFilter:
    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    with open(classifier_path, "rb") as model_file:
        classifier = pickle.load(model_file)

    return CropFilter(
        clip_model=clip_model,
        preprocess=preprocess,
        classifier=classifier,
        device=device,
        classes=DEFAULT_CLASSES,
    )


def predict_image_class(image_path: str, crop_filter: CropFilter | None = None) -> str:
    import torch
    from PIL import Image

    if crop_filter is None:
        crop_filter = load_crop_filter()

    image = crop_filter.preprocess(Image.open(image_path)).unsqueeze(0).to(crop_filter.device)

    with torch.no_grad():
        embedding = crop_filter.clip_model.encode_image(image)

    embedding = embedding.cpu().numpy()
    pred = crop_filter.classifier.predict(embedding)
    return crop_filter.classes[pred[0]]


def filter_machine_crops(
    crops_folder: str = "images",
    output_root: str = "dataset",
    move_files: bool = False,
) -> dict[str, int]:
    if not os.path.isdir(crops_folder):
        raise FileNotFoundError(f"Crops folder not found: {crops_folder}")

    crop_filter = load_crop_filter()

    os.makedirs(output_root, exist_ok=True)
    for class_name in DEFAULT_CLASSES:
        os.makedirs(os.path.join(output_root, class_name), exist_ok=True)

    counts = {"animated": 0, "real": 0, "skipped": 0}

    for file_name in sorted(os.listdir(crops_folder)):
        file_path = os.path.join(crops_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_name)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            counts["skipped"] += 1
            continue

        label = predict_image_class(file_path, crop_filter=crop_filter)
        label_folder = os.path.join(output_root, label)
        destination = os.path.join(label_folder, file_name)

        if move_files:
            shutil.move(file_path, destination)
        else:
            shutil.copy2(file_path, destination)

        counts[label] += 1

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify crops as animated or real")
    parser.add_argument("--image", help="Single image path to classify")
    parser.add_argument("--crops-folder", default="images", help="Folder with machine crops")
    parser.add_argument("--output-root", default="dataset", help="Root folder for animated/real outputs")
    parser.add_argument("--move-files", action="store_true", help="Move files instead of copying")
    args = parser.parse_args()

    if args.image:
        predicted = predict_image_class(args.image)
        print("Prediction:", predicted)
    else:
        stats = filter_machine_crops(
            crops_folder=args.crops_folder,
            output_root=args.output_root,
            move_files=args.move_files,
        )
        print("Filtering complete:", stats)