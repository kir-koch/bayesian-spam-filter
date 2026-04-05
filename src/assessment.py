from pathlib import Path
import pandas as pd


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _f1_score(precision, recall):
    return _safe_divide(2 * precision * recall, precision + recall)


def predict_split(model, split_df, split_name):
    rows = []
    for index, row in split_df.reset_index(drop=True).iterrows():
        pred_label, posteriors, _ = model.eval_message(row['message'])
        true_label_id = int(row['label'])
        true_label = 'spam' if true_label_id == 1 else 'ham'
        rows.append({
            'split': split_name,
            'row_id': index,
            'message': row['message'],
            'true_label_id': true_label_id,
            'true_label': true_label,
            'pred_label': pred_label,
            'ham_prob': float(posteriors[0]),
            'spam_prob': float(posteriors[1]),
            'is_correct': pred_label == true_label,
        })

    return pd.DataFrame(rows, columns=[
        'split',
        'row_id',
        'message',
        'true_label_id',
        'true_label',
        'pred_label',
        'ham_prob',
        'spam_prob',
        'is_correct',
    ])


def compute_metrics(pred_df, fit_on='unknown'):
    split_name = pred_df['split'].iloc[0] if len(pred_df) else 'unknown'
    total = int(len(pred_df))
    correct = int(pred_df['is_correct'].sum()) if total else 0
    incorrect = total - correct
    accuracy = (correct / total) if total else 0.0
    true_spam = pred_df['true_label'] == 'spam'
    pred_spam = pred_df['pred_label'] == 'spam'
    true_ham = pred_df['true_label'] == 'ham'
    pred_ham = pred_df['pred_label'] == 'ham'

    tp = int((true_spam & pred_spam).sum()) if total else 0
    tn = int((true_ham & pred_ham).sum()) if total else 0
    fp = int((true_ham & pred_spam).sum()) if total else 0
    fn = int((true_spam & pred_ham).sum()) if total else 0

    spam_precision = _safe_divide(tp, tp + fp)
    spam_recall = _safe_divide(tp, tp + fn)
    spam_f1 = _f1_score(spam_precision, spam_recall)

    ham_precision = _safe_divide(tn, tn + fn)
    ham_recall = _safe_divide(tn, tn + fp)
    ham_f1 = _f1_score(ham_precision, ham_recall)
    macro_f1 = (spam_f1 + ham_f1) / 2.0

    return {
        'split': split_name,
        'fit_on': fit_on,
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'spam_precision': spam_precision,
        'spam_recall': spam_recall,
        'spam_f1': spam_f1,
        'ham_precision': ham_precision,
        'ham_recall': ham_recall,
        'ham_f1': ham_f1,
        'macro_f1': macro_f1,
    }


def summarize_metrics(metrics_df):
    return metrics_df[[
        'split',
        'fit_on',
        'accuracy',
        'spam_precision',
        'spam_recall',
        'spam_f1',
        'fp',
        'fn',
    ]]


def assess_predictions(
    val_pred_df,
    test_pred_df,
    output_dir='artifacts',
    val_fit_on='train',
    test_fit_on='train+val',
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    val_pred_path = output_path / 'predictions_val.csv'
    test_pred_path = output_path / 'predictions_test.csv'
    metrics_path = output_path / 'metrics.csv'

    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    metrics_df = pd.DataFrame([
        compute_metrics(val_pred_df, fit_on=val_fit_on),
        compute_metrics(test_pred_df, fit_on=test_fit_on),
    ])
    metrics_df.to_csv(metrics_path, index=False)

    print('Assessment summary:')
    for metric in summarize_metrics(metrics_df).to_dict(orient='records'):
        split_label = 'validation' if metric['split'] == 'val' else metric['split']
        print(
            f"  {split_label} (fit on {metric['fit_on']}): "
            f"accuracy={metric['accuracy']:.4f} "
            f"spam_precision={metric['spam_precision']:.4f} "
            f"spam_recall={metric['spam_recall']:.4f} "
            f"spam_f1={metric['spam_f1']:.4f} "
            f"fp={int(metric['fp'])} fn={int(metric['fn'])}"
        )

    return metrics_df
