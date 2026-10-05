from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAW_FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
MODEL_FEATURES = ["GrLivArea", "BedroomAbvGr", "Bathrooms"]
TARGET = "SalePrice"


def create_demo_data(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    square_feet = np.clip(rng.normal(1650, 520, rows), 500, 4200).round()
    bedrooms = np.clip((square_feet / 650 + rng.normal(0, 0.8, rows)).round(), 1, 6)
    full_bath = np.clip((square_feet / 1100 + rng.normal(0.2, 0.6, rows)).round(), 1, 4)
    half_bath = rng.binomial(1, 0.35, rows)
    noise = rng.normal(0, 24000, rows)
    sale_price = (
        45000
        + square_feet * 112
        + bedrooms * 9000
        + full_bath * 18000
        + half_bath * 8000
        + noise
    )

    return pd.DataFrame(
        {
            "GrLivArea": square_feet,
            "BedroomAbvGr": bedrooms.astype(int),
            "FullBath": full_bath.astype(int),
            "HalfBath": half_bath.astype(int),
            "SalePrice": np.maximum(sale_price, 60000).round(0),
        }
    )


def load_dataset(path: Path, demo: bool) -> pd.DataFrame:
    if demo:
        print("Running in demo mode with generated housing data.")
        data = create_demo_data()
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download Kaggle train.csv into data/train.csv "
                "or run with --demo."
            )
        data = pd.read_csv(path)

    required = RAW_FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[required].copy()
    data["Bathrooms"] = data["FullBath"] + 0.5 * data["HalfBath"]
    return data[MODEL_FEATURES + [TARGET]]


def train_model(data: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = data[MODEL_FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": MODEL_FEATURES,
    }

    joblib.dump(model, output_dir / "linear_regression_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_actual_vs_predicted(y_test.to_numpy(), predictions, output_dir / "actual_vs_predicted.png")
    save_sample_predictions(x_test, y_test, predictions, output_dir / "sample_predictions.csv")
    return metrics


def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.65, edgecolor="k", linewidth=0.3)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    plt.plot([low, high], [low, high], color="crimson", linewidth=2)
    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_sample_predictions(
    x_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray, path: Path
) -> None:
    sample = x_test.copy()
    sample["ActualSalePrice"] = y_test.to_numpy()
    sample["PredictedSalePrice"] = predictions.round(0)
    sample["AbsoluteError"] = (sample["ActualSalePrice"] - sample["PredictedSalePrice"]).abs()
    sample.head(25).to_csv(path, index=False)


def predict_price(model_path: Path, square_feet: float, bedrooms: int, bathrooms: float) -> float:
    model = joblib.load(model_path)
    sample = pd.DataFrame(
        [{"GrLivArea": square_feet, "BedroomAbvGr": bedrooms, "Bathrooms": bathrooms}]
    )
    return float(model.predict(sample)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear regression for house price prediction.")
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--demo", action="store_true", help="Use generated data for a quick smoke test.")
    parser.add_argument("--square-feet", type=float, help="Square footage for an example prediction.")
    parser.add_argument("--bedrooms", type=int, help="Bedroom count for an example prediction.")
    parser.add_argument("--bathrooms", type=float, help="Bathroom count for an example prediction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data, args.demo)
    metrics = train_model(data, args.output_dir)

    print(json.dumps(metrics, indent=2))

    wants_prediction = all(
        value is not None for value in [args.square_feet, args.bedrooms, args.bathrooms]
    )
    if wants_prediction:
        predicted_price = predict_price(
            args.output_dir / "linear_regression_model.joblib",
            args.square_feet,
            args.bedrooms,
            args.bathrooms,
        )
        print(f"Predicted price: ${predicted_price:,.0f}")


if __name__ == "__main__":
    main()
