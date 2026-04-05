import argparse
from pathlib import Path

import pandas as pd

from src.assessment import assess_predictions, predict_split, summarize_metrics
from src.data_loader import parse_csv, split
from src.naive_bayes import NaiveBayes
from src.tfidf_logistic_regression import TfidfLogisticRegression
from src.tune_hyperparams import calculate_cost, coarse_to_fine, log_loss


DEFAULT_ALPHA_BOUNDS = [0.05, 20.0]
DEFAULT_C_BOUNDS = [0.05, 20.0]
DEFAULT_THRESHOLD_BOUNDS = [0.05, 0.95]


def positive_float(arg_name):
    def parse(value):
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{arg_name} must be a number')
        if value <= 0:
            raise argparse.ArgumentTypeError(f'{arg_name} must be greater than 0')
        return value

    return parse


def probability_float(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('spam threshold must be a number')
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError('spam threshold must be between 0 and 1')
    return value



def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate Bayesian spam filter.')
    parser.add_argument(
        '--model',
        choices=['naive_bayes', 'tfidf_logreg'],
        default='naive_bayes',
        help='Model to train and evaluate.',
    )
    parser.add_argument('--data-path', default='data/sms_spam_ham.csv', help='Path to training CSV data.')
    parser.add_argument('--output-dir', default='artifacts', help='Directory for assessment artifacts.')
    parser.add_argument(
        '--alpha',
        type=positive_float('alpha'),
        default=1.0,
        help='Laplace smoothing parameter.',
    )
    parser.add_argument(
        '--c',
        type=positive_float('c'),
        default=1.0,
        help='Inverse regularization strength for TF-IDF Logistic Regression.',
    )
    parser.add_argument(
        '--spam-threshold',
        type=probability_float,
        default=0.5,
        help='Decision threshold on P(spam) used by the final classifier.',
    )
    parser.add_argument(
        '--tune-alpha',
        action='store_true',
        help='Tune alpha on validation split using log loss.',
    )
    parser.add_argument(
        '--tune-c',
        action='store_true',
        help='Tune Logistic Regression C on validation split using log loss.',
    )
    parser.add_argument(
        '--tune-threshold',
        action='store_true',
        help='Tune spam threshold on validation split using weighted FP/FN cost.',
    )
    parser.add_argument(
        '--alpha-bounds',
        nargs=2,
        type=float,
        default=DEFAULT_ALPHA_BOUNDS,
        metavar=('LOW', 'HIGH'),
        help='Search interval for alpha tuning (must be > 0).',
    )
    parser.add_argument(
        '--c-bounds',
        nargs=2,
        type=float,
        default=DEFAULT_C_BOUNDS,
        metavar=('LOW', 'HIGH'),
        help='Search interval for Logistic Regression C tuning (must be > 0).',
    )
    parser.add_argument(
        '--threshold-bounds',
        nargs=2,
        type=float,
        default=DEFAULT_THRESHOLD_BOUNDS,
        metavar=('LOW', 'HIGH'),
        help='Search interval for threshold tuning (values in [0, 1]).',
    )
    parser.add_argument(
        '--cost-fp',
        type=float,
        default=1.0,
        help='Cost weight for false positives during threshold tuning.',
    )
    parser.add_argument(
        '--cost-fn',
        type=float,
        default=1.0,
        help='Cost weight for false negatives during threshold tuning.',
    )
    args = parser.parse_args()

    if args.model == 'naive_bayes':
        if args.tune_c:
            parser.error('--tune-c can only be used with --model tfidf_logreg')
        if args.c != 1.0:
            parser.error('--c can only be used with --model tfidf_logreg')
        if tuple(args.c_bounds) != tuple(DEFAULT_C_BOUNDS):
            parser.error('--c-bounds can only be used with --model tfidf_logreg')
    elif args.model == 'tfidf_logreg':
        if args.tune_alpha:
            parser.error('--tune-alpha can only be used with --model naive_bayes')
        if args.alpha != 1.0:
            parser.error('--alpha can only be used with --model naive_bayes')
        if tuple(args.alpha_bounds) != tuple(DEFAULT_ALPHA_BOUNDS):
            parser.error('--alpha-bounds can only be used with --model naive_bayes')

    return args


def get_model_spec(args):
    if args.model == 'naive_bayes':
        return {
            'display_name': 'Naive Bayes',
            'model_class': NaiveBayes,
            'primary_param_name': 'alpha',
            'primary_param_value': args.alpha,
            'primary_bounds': args.alpha_bounds,
            'tune_primary': args.tune_alpha,
            'primary_space': 'geometric',
        }

    return {
        'display_name': 'TF-IDF + Logistic Regression',
        'model_class': TfidfLogisticRegression,
        'primary_param_name': 'c',
        'primary_param_value': args.c,
        'primary_bounds': args.c_bounds,
        'tune_primary': args.tune_c,
        'primary_space': 'geometric',
    }


def main():
    args = parse_args()
    model_spec = get_model_spec(args)
    project_root = Path(__file__).resolve().parent
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    df = parse_csv(data_path)
    # Split once and keep test untouched until the final evaluation
    train_df, val_df, test_df = split(df)

    selected_primary = model_spec['primary_param_value']
    selected_threshold = args.spam_threshold
    tuning_records = []

    if model_spec['tune_primary']:
        selected_primary = coarse_to_fine(
            train_df,
            val_df,
            bounds=model_spec['primary_bounds'],
            param_name=model_spec['primary_param_name'],
            train_kwargs={'spam_threshold': selected_threshold},
            loss_funct=log_loss,
            model_class=model_spec['model_class'],
            space=model_spec['primary_space'],
        )
        tuning_records.append({
            'parameter': model_spec['primary_param_name'],
            'objective': 'log_loss',
            'selected_value': float(selected_primary),
        })
        print(
            f"Best {model_spec['primary_param_name']} from validation tuning: "
            f'{selected_primary:.6f}'
        )

    if args.tune_threshold:
        # Threshold objective trades off false positives vs false negatives
        threshold_loss = lambda model, split_df: calculate_cost(
            model,
            split_df,
            cost_fpos=args.cost_fp,
            cost_fneg=args.cost_fn,
        )
        selected_threshold = coarse_to_fine(
            train_df,
            val_df,
            bounds=args.threshold_bounds,
            param_name='spam_threshold',
            train_kwargs={model_spec['primary_param_name']: selected_primary},
            loss_funct=threshold_loss,
            model_class=model_spec['model_class'],
            space='linear',
        )
        tuning_records.append({
            'parameter': 'spam_threshold',
            'objective': f'weighted_cost(fp={args.cost_fp}, fn={args.cost_fn})',
            'selected_value': float(selected_threshold),
        })
        print(f'Best threshold from validation tuning: {selected_threshold:.6f}')

    selection_model = model_spec['model_class']()
    # Validation remains a clean selection stage fit only on the training split.
    selection_model.train(
        train_df,
        **{
            model_spec['primary_param_name']: selected_primary,
            'spam_threshold': selected_threshold,
        },
    )
    val_pred_df = predict_split(selection_model, val_df, 'val')

    final_train_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = model_spec['model_class']()
    # Refit on all non-test data before the final holdout evaluation.
    final_model.train(
        final_train_df,
        **{
            model_spec['primary_param_name']: selected_primary,
            'spam_threshold': selected_threshold,
        },
    )
    test_pred_df = predict_split(final_model, test_df, 'test')

    metrics_df = assess_predictions(
        val_pred_df,
        test_pred_df,
        output_dir=output_dir,
        val_fit_on='train',
        test_fit_on='train+val',
    )

    if tuning_records:
        tuning_df = pd.DataFrame(tuning_records)
        tuning_df.to_csv(output_dir / 'hyperparameter_tuning.csv', index=False)
        summary_df = summarize_metrics(metrics_df)
        print('\nSelected hyperparameters:')
        print(tuning_df.to_string(index=False))
        print('\nValidation metrics (used for model selection):')
        print(summary_df[summary_df['split'] == 'val'].to_string(index=False))
        print('\nTest metrics (final holdout estimate):')
        print(summary_df[summary_df['split'] == 'test'].to_string(index=False))
    else:
        print('\nEvaluation metrics:')
        print(summarize_metrics(metrics_df).to_string(index=False))
    print(f"\nModel: {model_spec['display_name']}")
    print(f"\nFinal {model_spec['primary_param_name']}: {selected_primary:.6f}")
    print(f'Final spam threshold: {selected_threshold:.6f}')
    print(f'\nArtifacts written to: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
