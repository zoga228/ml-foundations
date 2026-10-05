# PRODIGY_ML_05 - Food Recognition and Calorie Estimation

Task: develop a model that recognizes food items from images and estimates their calorie content.

Dataset: <https://www.kaggle.com/dansbecker/food-101>

## Run

Download and extract Food-101 into `data/`. The script can read class folders such as:

```text
data/food-101/images/apple_pie/*.jpg
data/food-101/images/pizza/*.jpg
```

Train a compact CNN:

```bash
python src/food_calorie_estimator.py --data-root data/food-101/images --epochs 8 --limit-per-class 250
```

For better real-world accuracy, use MobileNet:

```bash
python src/food_calorie_estimator.py --data-root data/food-101/images --model-type mobilenet --pretrained --epochs 6 --limit-per-class 300
```

Smoke-test without Kaggle data:

```bash
python src/food_calorie_estimator.py --demo --epochs 2
```

Predict one image:

```bash
python src/food_calorie_estimator.py --predict-image path/to/food.jpg --model outputs/food_calorie_model.pt
```

Calorie estimates use `calorie_reference.json` as approximate per-serving values. They are educational estimates, not dietary or medical advice.
