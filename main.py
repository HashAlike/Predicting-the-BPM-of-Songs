from src.data_loader import load_train_data, load_test_data
from config import TRAIN_PATH, TEST_PATH


train_df = load_train_data(TRAIN_PATH)
test_df = load_test_data(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)