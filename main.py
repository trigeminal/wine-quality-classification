import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.data_loader import load_data, preprocess_data
from src.model import wide_deep_model
from src.train import train_model
from src.config import EPOCHS, BATCH_SIZE, TEST_SIZE, RANDOM_STATE


def main():
    # Load and preprocess the data
    X, y = load_data("datasets/raw/white_wine.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Define and compile the model
    input_shape = X_train.shape[1]
    model = wide_deep_model(input_shape)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

    # Train the model
    train_model(
        model, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    main()
