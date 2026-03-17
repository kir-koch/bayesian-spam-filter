import numpy as np
from src.assessment import predict_split
from src.naive_bayes import NaiveBayes


def log_loss(model, df):
    # Binary cross-entropy on validation predictions
    eps = 1e-15
    pred_df = predict_split(model, df, 'val')

    y = pred_df['true_label_id'].to_numpy(dtype=float)
    p = pred_df['spam_prob'].to_numpy(dtype=float)
    p = np.clip(p, eps, 1.0 - eps)

    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(losses.mean())


# mode — tuned hyperparam: 'alpha' or 'spam_threshold'
# default — value for other hyperparam: float between 0 and 1
def coarse_to_fine(train_df, val_df, bounds, mode, default, loss_funct):
    """Iteratively shrink the search interval around the best grid point"""
    if len(bounds) != 2:
        raise ValueError('bounds must contain exactly two numbers: [start, end]')

    start, end = float(bounds[0]), float(bounds[1])
    if start >= end:
        raise ValueError('bounds must satisfy start < end')
    if mode == 'alpha' and start <= 0:
        raise ValueError('alpha bounds must be > 0 for geometric search')
    if mode == 'spam_threshold' and (start < 0 or end > 1):
        raise ValueError('spam_threshold bounds must be within [0, 1]')

    def optimize_interval(start, end, space):
        NUM_POINTS = 12
        grid = space(start, end, NUM_POINTS)
        opt_score = float('inf')
        best_bounds = [start, end]

        for i, hyperparam in enumerate(grid):
            model = NaiveBayes()
            if mode == 'alpha':
                model.train(train_df, hyperparam, default)
            elif mode == 'spam_threshold':
                model.train(train_df, default, hyperparam)

            cur_score = loss_funct(model, val_df)
            if cur_score < opt_score:
                opt_score = cur_score
                # Keep neighbors of the current best point as the next interval
                best_bounds[0] = grid[max(0, i - 1)]
                best_bounds[1] = grid[min(NUM_POINTS - 1, i + 1)]
        return best_bounds

    if mode == 'alpha':
        space = np.geomspace
    elif mode == 'spam_threshold':
        space = np.linspace
    else:
        raise ValueError("mode must be either 'alpha' or 'spam_threshold'")

    bounds = optimize_interval(start, end, space)
    for _ in range(3):
        bounds = optimize_interval(bounds[0], bounds[1], space)
    return np.mean(bounds)


def calculate_cost(model, df, cost_fpos, cost_fneg):
    """Weighted misclassification cost used for threshold tuning"""
    pred = predict_split(model, df, 'val')
    pred_false = pred[pred['is_correct'] == False]

    total = len(pred)
    if total == 0:
        return 0.0
    false_pos = len(pred_false[pred_false['pred_label'] == 'spam'])
    false_neg = len(pred_false) - false_pos

    cost = (cost_fpos * false_pos + cost_fneg * false_neg) / total
    return cost
