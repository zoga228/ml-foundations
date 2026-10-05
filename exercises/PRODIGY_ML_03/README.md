# PRODIGY_ML_03 - Cats vs Dogs SVM

Task: implement a support vector machine to classify cat and dog images from the Kaggle dataset.

Dataset: <https://www.kaggle.com/c/dogs-vs-cats/data>

## Run

Download the Kaggle files into `data/`. After extraction, the script can read either:

- `data/train/` containing `cat.0.jpg`, `dog.0.jpg`, ...
- `data/train.zip`

Train on a balanced subset:

```bash
python src/cat_dog_svm.py --data data/train --limit-per-class 2000
```

Smoke-test without Kaggle data:

```bash
python src/cat_dog_svm.py --demo --limit-per-class 120
```

Predict one image after training:

```bash
python src/cat_dog_svm.py --predict-image path/to/image.jpg --model outputs/cat_dog_svm.joblib
```
