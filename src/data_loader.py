import pandas as pd
from src.preprocessing import tokenize_text


def parse_csv(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df['tokens'] = df['message'].apply(tokenize_text)
    return df


def split(df, train_percentage=80, val_percentage=10):
    if min(train_percentage, val_percentage) < 1:
        raise ValueError('Enter values between 1% and 99%')
    if train_percentage + val_percentage >= 100:
        raise ValueError('The sum of train and validation percentages should be less than 100%')

    df_shuffled = df.sample(frac=1, random_state=1729).reset_index(drop=True)

    size = len(df)
    train_end = int(size * train_percentage / 100)
    val_end = train_end + int(size * val_percentage / 100)

    train_df, val_df, test_df = [
        df_shuffled.iloc[start:end].reset_index(drop=True)
        for start, end in (
            (None, train_end), 
            (train_end, val_end), 
            (val_end, None)
        )
    ]

    return train_df, val_df, test_df
