from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEMO_CLASSES = ["pizza", "sushi", "ice_cream", "greek_salad"]


class SmallFoodCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class FoodImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int) -> None:
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = make_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


class SyntheticFoodDataset(Dataset):
    def __init__(self, samples_per_class: int, image_size: int) -> None:
        self.classes = DEMO_CLASSES
        self.image_size = image_size
        self.items = [
            (class_index, sample_index)
            for class_index in range(len(self.classes))
            for sample_index in range(samples_per_class)
        ]
        self.targets = [class_index for class_index, _ in self.items]
        self.transform = make_transform(image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        class_index, sample_index = self.items[index]
        image = draw_demo_food(self.classes[class_index], sample_index, self.image_size)
        return self.transform(image), class_index


def make_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def draw_demo_food(label: str, index: int, size: int) -> Image.Image:
    rng = np.random.default_rng(index + 97 * DEMO_CLASSES.index(label))
    image = Image.new("RGB", (size, size), color=tuple(rng.integers(25, 70, size=3)))
    draw = ImageDraw.Draw(image)
    cx = size // 2 + int(rng.integers(-5, 6))
    cy = size // 2 + int(rng.integers(-5, 6))

    if label == "pizza":
        draw.ellipse((18, 18, size - 18, size - 18), fill=(226, 177, 79), outline=(170, 98, 35), width=4)
        draw.polygon([(cx, cy), (size - 20, 24), (size - 24, size - 24)], fill=(215, 70, 55))
        for _ in range(16):
            x, y = rng.integers(26, size - 26, size=2)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(120, 30, 30))
    elif label == "sushi":
        for row in range(2):
            for col in range(3):
                x = 24 + col * 28
                y = 34 + row * 30
                draw.ellipse((x, y, x + 22, y + 22), fill=(245, 245, 235), outline=(30, 30, 30), width=4)
                draw.ellipse((x + 7, y + 7, x + 15, y + 15), fill=(240, 90, 90))
    elif label == "ice_cream":
        draw.polygon([(cx - 26, cy + 10), (cx + 26, cy + 10), (cx, size - 12)], fill=(210, 150, 72))
        for offset, color in [(-18, (244, 190, 210)), (0, (245, 245, 225)), (18, (130, 80, 55))]:
            draw.ellipse((cx + offset - 18, cy - 35, cx + offset + 18, cy + 2), fill=color)
    else:
        draw.ellipse((18, 18, size - 18, size - 18), fill=(40, 120, 70), outline=(210, 230, 210), width=4)
        for _ in range(22):
            x, y = rng.integers(24, size - 24, size=2)
            color = tuple(int(value) for value in rng.choice([[220, 40, 40], [245, 245, 210], [70, 180, 85]]))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    return image


def normalize_food_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().lower()


def discover_samples(root: Path, limit_per_class: int | None) -> tuple[list[tuple[Path, int]], list[str]]:
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found at {root}. Download Food-101 or run with --demo."
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            grouped[path.parent.name].append(path)

    classes = sorted(class_name for class_name, files in grouped.items() if len(files) >= 2)
    if len(classes) < 2:
        raise ValueError("At least two food classes with images are required.")

    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for class_name in classes:
        files = sorted(grouped[class_name])
        if limit_per_class:
            files = files[:limit_per_class]
        samples.extend((file, class_to_index[class_name]) for file in files)
    return samples, classes


def build_model(model_type: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_type == "simple":
        return SmallFoodCNN(num_classes)

    if model_type == "mobilenet":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model type: {model_type}")


def make_loaders(dataset: Dataset, labels: list[int], batch_size: int) -> tuple[DataLoader, DataLoader]:
    indices = np.arange(len(labels))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    return (
        DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False),
    )


def train_model(
    dataset: Dataset,
    classes: list[str],
    calorie_reference: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(dataset, list(dataset.targets), args.batch_size)
    model = build_model(args.model_type, len(classes), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_total,
                "train_accuracy": train_correct / train_total,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch}: train_acc={history[-1]['train_accuracy']:.3f} "
            f"val_acc={val_accuracy:.3f}"
        )

    save_outputs(model, classes, calorie_reference, history, y_true, y_pred, args)
    return {
        "classes": classes,
        "epochs": args.epochs,
        "model_type": args.model_type,
        "validation_accuracy": float(history[-1]["val_accuracy"]),
        "train_images": int(len(train_loader.dataset)),
        "validation_images": int(len(val_loader.dataset)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


def save_outputs(
    model: nn.Module,
    classes: list[str],
    calorie_reference: dict[str, int],
    history: list[dict],
    y_true: list[int],
    y_pred: list[int],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "image_size": args.image_size,
            "model_type": args.model_type,
            "pretrained": args.pretrained,
        },
        args.output_dir / "food_calorie_model.pt",
    )
    (args.output_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (args.output_dir / "calorie_reference_used.json").write_text(
        json.dumps(calorie_reference, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, zero_division=0), encoding="utf-8"
    )
    plot_history(history, args.output_dir / "training_history.png")
    plot_confusion_matrix(y_true, y_pred, classes, args.output_dir / "confusion_matrix.png")


def plot_history(history: list[dict], path: Path) -> None:
    frame = pd.DataFrame(history)
    plt.figure(figsize=(7, 5))
    plt.plot(frame["epoch"], frame["train_accuracy"], marker="o", label="train")
    plt.plot(frame["epoch"], frame["val_accuracy"], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Food Classifier Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], classes: list[str], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    size = max(6, min(18, len(classes) * 0.45))
    plt.figure(figsize=(size, size))
    sns.heatmap(matrix, annot=False, cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Food Recognition Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def load_calorie_reference(path: Path) -> dict[str, int]:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {normalize_food_name(name): int(calories) for name, calories in raw.items()}
    return {}


def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int, dict]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])
    image_size = int(checkpoint["image_size"])
    model_type = str(checkpoint.get("model_type", "simple"))
    pretrained = bool(checkpoint.get("pretrained", False))
    model = build_model(model_type, len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, image_size, checkpoint


def predict_image(model_path: Path, image_path: Path, calorie_reference: dict[str, int], device: torch.device) -> dict:
    model, classes, image_size, _ = load_model(model_path, device)
    transform = make_transform(image_size)
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    index = int(probabilities.argmax().item())
    class_name = classes[index]
    normalized = normalize_food_name(class_name)
    calories = calorie_reference.get(normalized)
    return {
        "food": normalized,
        "confidence": float(probabilities[index].item()),
        "estimated_calories_per_serving": calories if calories is not None else "unknown",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food recognition and calorie estimation.")
    parser.add_argument("--data-root", type=Path, default=Path("data/food-101/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--calorie-reference", type=Path, default=Path("calorie_reference.json"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--model-type", choices=["simple", "mobilenet"], default="simple")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights with MobileNet.")
    parser.add_argument("--demo", action="store_true", help="Use generated food images for a quick smoke test.")
    parser.add_argument("--demo-samples-per-class", type=int, default=80)
    parser.add_argument("--predict-image", type=Path, help="Classify one image using a saved model.")
    parser.add_argument("--model", type=Path, default=Path("outputs/food_calorie_model.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calorie_reference = load_calorie_reference(args.calorie_reference)

    if args.predict_image:
        result = predict_image(args.model, args.predict_image, calorie_reference, device)
        print(json.dumps(result, indent=2))
        return

    if args.demo:
        print("Running in demo mode with generated food images.")
        dataset = SyntheticFoodDataset(args.demo_samples_per_class, args.image_size)
        classes = dataset.classes
    else:
        samples, classes = discover_samples(args.data_root, args.limit_per_class)
        dataset = FoodImageDataset(samples, args.image_size)

    metrics = train_model(dataset, classes, calorie_reference, args, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
