import argparse
from pathlib import Path

import pandas as pd

from src.assessment import assess_model
from src.data_loader import parse_csv, split
from src.naive_bayes import NaiveBayes
from src.tune_hyperparams import calculate_cost, coarse_to_fine, log_loss


DEFAULT_ALPHA_BOUNDS = [0.05, 20.0]
DEFAULT_THRESHOLD_BOUNDS = [0.05, 0.95]


def positive_float(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('alpha must be a number')
    if value <= 0:
        raise argparse.ArgumentTypeError('alpha must be greater than 0')



def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate Bayesian spam filter.')
    parser.add_argument('--data-path', default='data/spam.csv', help='Path to training CSV data.')
    parser.add_argument('--output-dir', default='artifacts', help='Directory for assessment artifacts.')
    parser.add_argument('--alpha', type=positive_float, default=1.0, help='Laplace smoothing parameter.')
    parser.add_argument(
        '--spam-threshold',
        type=float,
        default=0.5,
        help='Decision threshold on P(spam) used by the final classifier.',
    )
    parser.add_argument(
        '--tune-alpha',
        action='store_true',
        help='Tune alpha on validation split using log loss.',
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
    return parser.parse_args()


def main():
    args = parse_args()
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

    selected_alpha = args.alpha
    selected_threshold = args.spam_threshold
    tuning_records = []

    if args.tune_alpha:
        selected_alpha = coarse_to_fine(
            train_df,
            val_df,
            bounds=args.alpha_bounds,
            mode='alpha',
            default=selected_threshold,
            loss_funct=log_loss,
        )
        tuning_records.append({
            'parameter': 'alpha',
            'objective': 'log_loss',
            'selected_value': float(selected_alpha),
        })
        print(f'Best alpha from validation tuning: {selected_alpha:.6f}')

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
            mode='spam_threshold',
            default=selected_alpha,
            loss_funct=threshold_loss,
        )
        tuning_records.append({
            'parameter': 'spam_threshold',
            'objective': f'weighted_cost(fp={args.cost_fp}, fn={args.cost_fn})',
            'selected_value': float(selected_threshold),
        })
        print(f'Best threshold from validation tuning: {selected_threshold:.6f}')

    model = NaiveBayes()
    # Final fit uses the selected hyperparameters from validation tuning
    model.train(train_df, alpha=selected_alpha, spam_threshold=selected_threshold)

    metrics_df = assess_model(model, val_df, test_df, output_dir=output_dir)

    if tuning_records:
        tuning_df = pd.DataFrame(tuning_records)
        tuning_df.to_csv(output_dir / 'hyperparameter_tuning.csv', index=False)
        print('\nSelected hyperparameters:')
        print(tuning_df.to_string(index=False))

    print('\nFinal metrics:')
    print(metrics_df.to_string(index=False))
    print(f'\nFinal alpha: {selected_alpha:.6f}')
    print(f'Final spam threshold: {selected_threshold:.6f}')
    print(f'\nArtifacts written to: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
