# Bayesian Spam Filter

A multinomial Naive Bayes classifier built in Python with numpy, pandas, and nltk. Trains a model on `data/spam.csv`, supports hyperparameter tuning, and exports prediction/metrics artifacts for analysis.


## Main features
- Text preprocessing with lowercasing, punctuation removal, stopword filtering, and stemming
- Data split: 80%/10%/10% — train/validation/test (random_seed=1729)
- Naive Bayes classifier supports:
  - Laplace smoothing (`alpha`)
  - Adjustable spam decision threshold (`spam_threshold`)
- Hyperparameter tuning on validation split:
  - `alpha` tuned with log loss
  - `spam_threshold` tuned with weighted FP/FN cost
- CSV artifact export for validation/test predictions and metrics


## Project Structure
- `main.py`: CLI entry point for training, tuning, and evaluation
- `src/preprocessing.py`: tokenization/stemming/stopword logic
- `src/data_loader.py`: dataset parsing and train/val/test split
- `src/naive_bayes.py`: model training and prediction
- `src/tune_hyperparams.py`: objective functions and coarse-to-fine search
- `src/assessment.py`: prediction tables, metrics computation, artifact writing


## Setup

```bash
cd /path/to/naive-bayes-spam
python3 -m pip install -r requirements.txt
```

Download NLTK stopwords once:
```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

## Run
Default training/evaluation:

```bash
python3 main.py
```

Tune `alpha` and threshold on validation data:

```bash
python3 main.py --tune-alpha --tune-threshold --output-dir artifacts_tuned
```

Tune with custom false positive/false negative costs:

```bash
python3 main.py --tune-alpha --tune-threshold --cost-fp 5 --cost-fn 1 --output-dir artifacts_cost_5_1
```

Manual hyperparameters (no tuning):

```bash
python3 main.py --alpha 0.2 --spam-threshold 0.8 --output-dir artifacts_manual
```

## Artifacts
Each run writes files to `--output-dir`:
- `predictions_val.csv`
- `predictions_test.csv`
- `metrics.csv`
- `hyperparameter_tuning.csv` (only when tuning is enabled)

Contents:
- `predictions_*.csv`:
  - true/predicted labels
  - class probabilities
  - `is_correct`
- `metrics.csv`:
  - split
  - number of correct/incorrect predictions
  - accuracy
- `hyperparameter_tuning.csv`: 
  - selected `alpha`/`spam_threshold`
  - validation/test accuracies


## Possible Next Improvements
- Add character and word n-grams, plus message length
- Implement TF-IDF + Logistic Regression
- Add cross-validation for more robust tuning
