import sys
import pandas as pd
import pandas as pd
import numpy as np

def split_train_backtest(df, train_ratio=0.2, random_state=None):
    """
    Splits the DataFrame into training and backtest sets based on a ratio.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        train_ratio (float): The ratio of the training set (default is 0.8).
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame, pd.DataFrame: training and backtest DataFrames.
    """
    shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_size = int(len(df) * train_ratio)
    train_df = shuffled_df.iloc[:train_size]
    backtest_df = shuffled_df.iloc[train_size:]
    return train_df, backtest_df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        train, backtest = split_train_backtest(df)
        train.to_csv("training.csv", index=0)
        backtest.to_csv("backtest.csv", index=0)
