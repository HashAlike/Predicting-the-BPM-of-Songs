from sklearn.model_selection import train_test_split


def prepare_data(df):
    X = df.drop(columns=["BeatsPerMinute", "id"])
    y = df["BeatsPerMinute"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_valid, y_train, y_valid


def prepare_test_data(df):
    return df.drop(columns=["id"])