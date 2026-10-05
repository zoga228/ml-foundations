# PRODIGY_ML_01 - House Price Prediction

Task: implement a linear regression model to predict house prices using square footage, bedrooms, and bathrooms.

Dataset: <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>

## Run

Download `train.csv` from Kaggle into `data/train.csv`, then run:

```bash
python src/house_price_regression.py --data data/train.csv
```

Smoke-test without Kaggle data:

```bash
python src/house_price_regression.py --demo
```

The script saves the trained model, metrics, and an actual-vs-predicted chart in `outputs/`.

## Example Prediction

```bash
python src/house_price_regression.py --data data/train.csv --square-feet 1800 --bedrooms 3 --bathrooms 2.5
```
