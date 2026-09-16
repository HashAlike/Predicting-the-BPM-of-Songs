import pandas as pd


def load_train_data(path):
    return pd.read_csv(path)


def load_test_data(path):
    return pd.read_csv(path)