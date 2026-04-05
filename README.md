# Bayesian Spam Filter

I built this project to better understand how simple statistical ideas can be used for spam classification in Python. It compares two approaches on the same SMS dataset:

- `Naive Bayes`, implemented from scratch
- `TF-IDF + Logistic Regression`, built with `scikit-learn`

Both models train on `data/sms_spam_ham.csv`, use the same text preprocessing, can be tuned on a validation split, and write prediction/metrics files for later inspection.

Current verified results on a single fixed stratified split (`random_state=1729`):

| Model | Validation accuracy | Test accuracy | Test spam precision | Test spam recall | Test spam F1 | Test FP | Test FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes tuned (`alpha=0.871361`, `spam_threshold=0.676945`) | `97.48%` | `98.21%` | `0.9583` | `0.9079` | `0.9324` | `3` | `7` |
| TF-IDF + Logistic Regression tuned (`c=19.995909`, `spam_threshold=0.249044`) | `97.84%` | `98.39%` | `0.9467` | `0.9342` | `0.9404` | `4` | `5` |

Validation metrics are computed with a model trained on the training split only. The final test metric is computed after retraining with `train + validation`, while keeping the test split untouched until the last step.

## Model Comparison

On this split, `TF-IDF + Logistic Regression` does slightly better than `Naive Bayes` on the final test set. The difference is not huge, but it is consistent across the most useful summary metrics:

- higher test accuracy: `98.39%` vs `98.21%`
- higher spam recall: `0.9342` vs `0.9079`
- higher spam F1: `0.9404` vs `0.9324`

The main tradeoff is that Logistic Regression allows one extra false positive (`4` vs `3`) while reducing false negatives (`5` vs `7`). In simple terms, it catches more spam, but it is also a little more willing to call a message spam.

Why it likely helps:

- `TF-IDF` gives less weight to very common words and more weight to words that carry more useful information
- `Logistic Regression` can learn how strongly different words should influence the decision
- this helps when spam is signaled not by one word alone, but by a combination of cues

## What The Project Includes
- Two model options via `--model`:
  - `naive_bayes`
  - `tfidf_logreg`
- Shared text preprocessing with lowercasing, punctuation removal, stopword filtering, and stemming
- Single fixed stratified 80%/10%/10% train/validation/test split (`random_state=1729`)
- Naive Bayes supports:
  - Laplace smoothing (`alpha`) with a shared training vocabulary
  - Adjustable spam decision threshold (`spam_threshold`)
- TF-IDF + Logistic Regression supports:
  - TF-IDF features built from the same tokenization pipeline
  - Logistic regression regularization strength (`c`)
  - Adjustable spam decision threshold (`spam_threshold`)
- Validation-based tuning:
  - `alpha` tuned with log loss
  - `c` tuned with log loss
  - `spam_threshold` tuned with weighted FP/FN cost
- Final refit on `train + validation` before the final test evaluation
- Safer CLI validation for `alpha` and `spam_threshold`
- CSV artifact export for validation/test predictions and metrics


## Project Structure
- `main.py`: CLI entry point for training, tuning, and evaluation
- `src/preprocessing.py`: tokenization/stemming/stopword logic
- `src/data_loader.py`: dataset parsing and train/val/test split
- `src/naive_bayes.py`: multinomial Naive Bayes training and prediction
- `src/tfidf_logistic_regression.py`: TF-IDF + Logistic Regression training and prediction
- `src/tune_hyperparams.py`: objective functions and coarse-to-fine search
- `src/assessment.py`: prediction tables, metrics computation, artifact writing


## Setup

```bash
cd /path/to/bayesian-spam-filter
python3 -m pip install -r requirements.txt
```

Download NLTK stopwords once:
```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

## Run
Default training/evaluation:

```bash
python3 main.py --output-dir artifacts_default
```

Tune Naive Bayes `alpha` and threshold on validation data:

```bash
python3 main.py --tune-alpha --tune-threshold --output-dir artifacts_tuned
```

Default TF-IDF + Logistic Regression run:

```bash
python3 main.py --model tfidf_logreg --output-dir artifacts_tfidf_logreg_default
```

Tune Logistic Regression `c` and threshold on validation data:

```bash
python3 main.py --model tfidf_logreg --tune-c --tune-threshold --output-dir artifacts_tfidf_logreg_tuned
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
  - training data used for that evaluation row (`fit_on`)
  - number of correct/incorrect predictions
  - accuracy
  - mistake counts (`tp`, `tn`, `fp`, `fn`)
  - spam-class precision / recall / F1
  - ham-class precision / recall / F1
  - macro-F1
- `hyperparameter_tuning.csv`: 
  - tuned parameter name
  - optimization objective
  - selected value

Validation metrics are used for model selection. Test metrics are reported separately as the final estimate after retraining on `train + validation`.

Terminal output stays concise by showing accuracy, the spam-class precision / recall / F1 summary, and the most important error counts (`fp`, `fn`).

Clean post-fix example outputs in this repo are written to:
- `artifacts_default/`
- `artifacts_tuned/`
- `artifacts_tfidf_logreg_default/`
- `artifacts_tfidf_logreg_tuned/`

## Error Analysis and Limitations

After reviewing the tuned test-set mistakes by hand, a few recurring patterns stand out.

Common failure patterns:

- Quote-like or content-light spam can fool both models when the message looks more like a generic statement than a promotion. Examples include the “driving slower than you” message and the horoscope-style `ROMCAPspam` message.
- Spam that imitates a normal personal conversation is also harder than obvious promotional spam. The “CLAIRE here ... Chat now ...” example partly reads like ordinary SMS dialogue before the premium-number pattern becomes clear.
- Surface words such as `winning` or `free` can trigger false positives even in legitimate messages. A ham message saying “this will increase the chance of winning” is classified as spam by both tuned models.

Model-specific observations:

- `Naive Bayes` is more vulnerable to non-English or code-mixed ham messages. In the current test split, it falsely flags two personal messages written in non-standard / non-English text that Logistic Regression handles correctly.
- `TF-IDF + Logistic Regression` is more aggressive on very short conversational ham messages such as “When you get free, call me” or “Txt bak luv K”. It appears to overreact to sparse spam-associated tokens like `call`, `text`, or `free` when there is very little surrounding context.
- `TF-IDF + Logistic Regression` does a better job on some personal-style spam that still contains a clear promotional pattern, such as premium-number or reply-based messages. That likely explains part of its better spam recall and F1.

Project limitations:

- The dataset is relatively small and focused on SMS-style messages, so results may not transfer directly to real email traffic.
- The current report is based on one fixed stratified split rather than cross-validation, so the measured advantage of one model over another should be treated as useful evidence, not as a final truth.
- Threshold tuning currently uses a neutral baseline cost setup. That is useful for comparison, but it does not yet reflect a real mail product where false positives can be more expensive than false negatives.
- The preprocessing pipeline is intentionally simple. It does not yet use n-grams, metadata, sender-level signals, or richer message structure.

## Possible Next Improvements
- Add character and word n-grams, plus message length
- Add cross-validation for more robust tuning
