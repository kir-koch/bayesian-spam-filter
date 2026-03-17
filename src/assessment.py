from pathlib import Path
import pandas as pd
from src.naive_bayes import NaiveBayes


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


def compute_metrics(pred_df):
    split_name = pred_df['split'].iloc[0] if len(pred_df) else 'unknown'
    total = int(len(pred_df))
    correct = int(pred_df['is_correct'].sum()) if total else 0
    incorrect = total - correct
    accuracy = (correct / total) if total else 0.0

    return {
        'split': split_name,
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
    }


def assess_model(model, val_df, test_df, output_dir='artifacts'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    val_pred_df = predict_split(model, val_df, 'val')
    test_pred_df = predict_split(model, test_df, 'test')

    val_pred_path = output_path / 'predictions_val.csv'
    test_pred_path = output_path / 'predictions_test.csv'
    metrics_path = output_path / 'metrics.csv'

    val_pred_df.to_csv(val_pred_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)

    metrics_df = pd.DataFrame([
        compute_metrics(val_pred_df),
        compute_metrics(test_pred_df),
    ])
    metrics_df.to_csv(metrics_path, index=False)

    print('Assessment summary:')
    for metric in metrics_df.to_dict(orient='records'):
        print(
            f"  {metric['split']}: accuracy={metric['accuracy']:.4f} "
            f"({metric['correct']}/{metric['total']})"
        )

    return metrics_df


def tune_alpha(train_df, val_df, alphas=None, output_dir='artifacts'):
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

    records = []
    for alpha in alphas:
        model = NaiveBayes()
        model.train(train_df, alpha=alpha)
        val_pred_df = predict_split(model, val_df, 'val')
        val_metrics = compute_metrics(val_pred_df)
        records.append({
            'alpha': float(alpha),
            'val_accuracy': float(val_metrics['accuracy']),
        })

    tuning_df = pd.DataFrame(records)
    ranked_df = tuning_df.sort_values(
        by=['val_accuracy', 'alpha'],
        ascending=[False, True],
    ).reset_index(drop=True)
    best_alpha = float(ranked_df.iloc[0]['alpha'])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(output_path / 'alpha_tuning.csv', index=False)

    return best_alpha, tuning_df
