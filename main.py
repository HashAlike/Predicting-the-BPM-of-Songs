from src.data_loader import load_train_data, load_test_data
from src.preprocessing import prepare_data, prepare_test_data
from src.model import create_gradient_boosting_model, train_model
from src.evaluate import evaluate_model

from config import TRAIN_PATH, TEST_PATH


# --------------------
# 1. Load data
# --------------------

train_df = load_train_data(TRAIN_PATH)
test_df = load_test_data(TEST_PATH)


# --------------------
# 2. Validation split
# --------------------

X_train, X_valid, y_train, y_valid = prepare_data(train_df)


# --------------------
# 3. Train baseline/final model
# --------------------

model = create_gradient_boosting_model()
model = train_model(model, X_train, y_train)


# --------------------
# 4. Evaluate
# --------------------

results = evaluate_model(
    model,
    X_valid,
    y_valid
)

print("Validation Results")
print(f"MAE: {results['MAE']:.4f}")
print(f"RMSE: {results['RMSE']:.4f}")
print(f"R²: {results['R2']:.4f}")


# --------------------
# 5. Train on full data
# --------------------

X_full = train_df.drop(columns=["BeatsPerMinute", "id"])
y_full = train_df["BeatsPerMinute"]

final_model = create_gradient_boosting_model()
final_model = train_model(final_model, X_full, y_full)


# --------------------
# 6. Prepare test data
# --------------------

X_test = prepare_test_data(test_df)


# --------------------
# 7. Predict
# --------------------

test_predictions = final_model.predict(X_test)


print("\nTest Predictions")
print(f"Min: {test_predictions.min():.4f}")
print(f"Max: {test_predictions.max():.4f}")
print(f"Mean: {test_predictions.mean():.4f}")



test_predictions = final_model.predict(X_test)

print("\nTest Predictions")
print(f"Min: {test_predictions.min():.4f}")
print(f"Max: {test_predictions.max():.4f}")
print(f"Mean: {test_predictions.mean():.4f}")




submission = test_df[["id"]].copy()
submission["BeatsPerMinute"] = test_predictions

submission.to_csv(
    "data/submission.csv",
    index=False
)

print("\nSubmission created: data/submission.csv")