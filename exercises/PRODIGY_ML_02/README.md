# PRODIGY_ML_02 - Customer Segmentation

Task: create a K-means clustering algorithm to group retail-store customers based on purchase behavior.

Dataset: <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>

## Run

Download `Mall_Customers.csv` from Kaggle into `data/Mall_Customers.csv`, then run:

```bash
python src/customer_segmentation.py --data data/Mall_Customers.csv --clusters 5
```

Smoke-test without Kaggle data:

```bash
python src/customer_segmentation.py --demo
```

The script saves the clustered customers, trained K-means model, cluster summary, elbow plot, silhouette plot, and a 2D cluster visualization in `outputs/`.
