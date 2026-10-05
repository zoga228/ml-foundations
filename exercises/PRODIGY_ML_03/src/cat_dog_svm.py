from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


LABELS = ["cat", "dog"]


def infer_label(name: str) -> str | None:
    lower = Path(name).name.lower()
    if lower.startswith("cat."):
        return "cat"
    if lower.startswith("dog."):
        return "dog"
    return None


def extract_hog_features(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB").resize((image_size, image_size))
    gray = rgb2gray(np.asarray(image))
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def read_image_from_zip(zip_file: zipfile.ZipFile, member: str) -> Image.Image:
    with zip_file.open(member) as stream:
        return Image.open(io.BytesIO(stream.read())).convert("RGB")


def collect_real_samples(path: Path, limit_per_class: int | None) -> list[tuple[str, str]]:
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if infer_label(name)]
        return balance_names([(name, infer_label(name)) for name in names if infer_label(name)], limit_per_class)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Extract Kaggle train.zip into data/train, "
            "or run with --demo."
        )

    image_paths = [
        str(file)
        for file in path.rglob("*")
        if file.suffix.lower() in {".jpg", ".jpeg", ".png"} and infer_label(file.name)
    ]
    return balance_names([(name, infer_label(name)) for name in image_paths if infer_label(name)], limit_per_class)


def balance_names(samples: list[tuple[str, str]], limit_per_class: int | None) -> list[tuple[str, str]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, label in samples:
        if label:
            grouped[label].append((name, label))

    balanced: list[tuple[str, str]] = []
    for label in LABELS:
        names = sorted(grouped[label])
        if limit_per_class:
            names = names[:limit_per_class]
        balanced.extend(names)
    if not all(grouped[label] for label in LABELS):
        raise ValueError("Both cat and dog images are required.")
    return balanced


def load_real_features(path: Path, limit_per_class: int | None, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    samples = collect_real_samples(path, limit_per_class)
    features: list[np.ndarray] = []
    labels: list[str] = []

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member, label in tqdm(samples, desc="Extracting HOG"):
                features.append(extract_hog_features(read_image_from_zip(archive, member), image_size))
                labels.append(label)
    else:
        for file_name, label in tqdm(samples, desc="Extracting HOG"):
            with Image.open(file_name) as image:
                features.append(extract_hog_features(image, image_size))
            labels.append(label)

    return np.vstack(features), np.array(labels)


def create_demo_image(label: str, index: int, size: int) -> Image.Image:
    rng = np.random.default_rng(index + (0 if label == "cat" else 10_000))
    background = rng.integers(15, 65, size=(size, size, 3), dtype=np.uint8)
    image = Image.fromarray(background)
    draw = ImageDraw.Draw(image)

    if label == "cat":
        draw.ellipse((18, 24, size - 18, size - 12), outline=(240, 210, 120), width=4)
        draw.polygon([(28, 26), (38, 8), (48, 28)], fill=(240, 210, 120))
        draw.polygon([(size - 48, 28), (size - 38, 8), (size - 28, 26)], fill=(240, 210, 120))
        draw.line((20, size - 26, size - 20, size - 26), fill=(255, 255, 255), width=2)
    else:
        draw.rounded_rectangle((18, 18, size - 18, size - 14), radius=14, outline=(130, 205, 255), width=5)
        draw.ellipse((10, 28, 34, 56), fill=(130, 205, 255))
        draw.ellipse((size - 34, 28, size - 10, 56), fill=(130, 205, 255))
        draw.rectangle((size // 2 - 16, size - 32, size // 2 + 16, size - 22), fill=(130, 205, 255))
    return image


def load_demo_features(limit_per_class: int | None, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    per_class = limit_per_class or 120
    features: list[np.ndarray] = []
    labels: list[str] = []
    for label in LABELS:
        for index in range(per_class):
            features.append(extract_hog_features(create_demo_image(label, index, image_size), image_size))
            labels.append(label)
    return np.vstack(features), np.array(labels)


def train_svm(x: np.ndarray, y: np.ndarray, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=8000)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "train_images": int(len(x_train)),
        "test_images": int(len(x_test)),
        "labels": LABELS,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0), encoding="utf-8"
    )
    joblib.dump(model, output_dir / "cat_dog_svm.joblib")
    plot_confusion_matrix(y_test, predictions, output_dir / "confusion_matrix.png")
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Cats vs Dogs SVM")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def predict_image(model_path: Path, image_path: Path, image_size: int) -> str:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    model = joblib.load(model_path)
    with Image.open(image_path) as image:
        features = extract_hog_features(image, image_size).reshape(1, -1)
    return str(model.predict(features)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cats vs dogs classification with HOG + linear SVM.")
    parser.add_argument("--data", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=80)
    parser.add_argument("--limit-per-class", type=int, default=1000)
    parser.add_argument("--demo", action="store_true", help="Use generated images for a quick smoke test.")
    parser.add_argument("--predict-image", type=Path, help="Classify one image using a saved model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/cat_dog_svm.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_image:
        label = predict_image(args.model, args.predict_image, args.image_size)
        print(f"Predicted label: {label}")
        return

    if args.demo:
        print("Running in demo mode with generated image patterns.")
        x, y = load_demo_features(args.limit_per_class, args.image_size)
    else:
        x, y = load_real_features(args.data, args.limit_per_class, args.image_size)

    metrics = train_svm(x, y, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
