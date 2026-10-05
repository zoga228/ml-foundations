from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def create_demo_data(rows: int = 260, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [24, 28, 78],
            [28, 74, 82],
            [46, 55, 48],
            [54, 86, 20],
            [61, 35, 35],
        ]
    )
    cluster_ids = rng.integers(0, len(centers), rows)
    noise = rng.normal([0, 0, 0], [5, 9, 10], size=(rows, 3))
    values = centers[cluster_ids] + noise
    values[:, 0] = np.clip(values[:, 0], 18, 70)
    values[:, 1] = np.clip(values[:, 1], 15, 140)
    values[:, 2] = np.clip(values[:, 2], 1, 100)

    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, rows + 1),
            "Gender": rng.choice(["Male", "Female"], rows),
            "Age": values[:, 0].round().astype(int),
            "Annual Income (k$)": values[:, 1].round().astype(int),
            "Spending Score (1-100)": values[:, 2].round().astype(int),
        }
    )


def load_dataset(path: Path, demo: bool) -> pd.DataFrame:
    if demo:
        print("Running in demo mode with generated customer data.")
        data = create_demo_data()
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download Mall_Customers.csv into data/ "
                "or run with --demo."
            )
        data = pd.read_csv(path)

    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data.copy()


def evaluate_k_range(x: pd.DataFrame, output_dir: Path, max_k: int) -> list[dict]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    rows: list[dict] = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        labels = kmeans.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(x_scaled, labels)),
            }
        )

    scores = pd.DataFrame(rows)
    scores.to_csv(output_dir / "k_selection_scores.csv", index=False)
    plot_k_selection(scores, output_dir)
    return rows


def fit_kmeans(data: pd.DataFrame, clusters: int, output_dir: Path, max_k: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = data[FEATURES]
    k_scores = evaluate_k_range(x, output_dir, max_k=max_k)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=clusters, init="k-means++", n_init=30, random_state=42)),
        ]
    )
    labels = pipeline.fit_predict(x)

    clustered = data.copy()
    clustered["Cluster"] = labels
    clustered.to_csv(output_dir / "clustered_customers.csv", index=False)
    joblib.dump(pipeline, output_dir / "kmeans_customer_segments.joblib")

    summary = (
        clustered.groupby("Cluster")[FEATURES]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    summary.to_csv(output_dir / "cluster_summary.csv")
    plot_clusters(clustered, output_dir / "customer_segments.png")

    best_k = max(k_scores, key=lambda row: row["silhouette"])["k"]
    metrics = {
        "chosen_clusters": clusters,
        "best_k_by_silhouette": int(best_k),
        "rows": int(len(clustered)),
        "features": FEATURES,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def plot_k_selection(scores: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="seagreen")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(output_dir / "k_selection.png", dpi=160)
    plt.close(fig)


def plot_clusters(clustered: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=clustered,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="tab10",
        s=70,
        edgecolor="white",
        linewidth=0.6,
    )
    plt.title("Customer Segments by Income and Spending Score")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-means customer segmentation.")
    parser.add_argument("--data", type=Path, default=Path("data/Mall_Customers.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--demo", action="store_true", help="Use generated data for a quick smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2")
    if args.max_k < args.clusters:
        raise ValueError("--max-k must be greater than or equal to --clusters")

    data = load_dataset(args.data, args.demo)
    metrics = fit_kmeans(data, args.clusters, args.output_dir, args.max_k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
