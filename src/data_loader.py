import pandas as pd
from src.preprocessing import tokenize_text


DEFAULT_RANDOM_SEED = 1729


def parse_csv(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df['tokens'] = df['message'].apply(tokenize_text)
    return df


def _split_one_frame(df, train_percentage, val_percentage):
    size = len(df)
    train_end = int(size * train_percentage / 100)
    val_end = train_end + int(size * val_percentage / 100)

    train_df, val_df, test_df = [
        df.iloc[start:end].reset_index(drop=True)
        for start, end in (
            (None, train_end),
            (train_end, val_end),
            (val_end, None),
        )
    ]
    return train_df, val_df, test_df


def split(df, train_percentage=80, val_percentage=10, random_seed=DEFAULT_RANDOM_SEED):
    if min(train_percentage, val_percentage) < 1:
        raise ValueError('Enter values between 1% and 99%')
    if train_percentage + val_percentage >= 100:
        raise ValueError('The sum of train and validation percentages should be less than 100%')

    train_parts = []
    val_parts = []
    test_parts = []

    for label, group_df in df.groupby('label', sort=True):
        group_shuffled = group_df.sample(
            frac=1,
            random_state=random_seed + int(label),
        ).reset_index(drop=True)
        group_train, group_val, group_test = _split_one_frame(
            group_shuffled,
            train_percentage,
            val_percentage,
        )
        train_parts.append(group_train)
        val_parts.append(group_val)
        test_parts.append(group_test)

    train_df = pd.concat(train_parts, ignore_index=True).sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(
        frac=1,
        random_state=random_seed,
    ).reset_index(drop=True)

    return train_df, val_df, test_df
