import warnings
import numpy as np
from src.data_loader import load_data, preprocess_data
from src.model import lightgbm_model
from src.train import train_model
from src.config import TEST_SIZE, RANDOM_STATE

warnings.filterwarnings("ignore")


# Best hyperparameters for LightGBM
best_params = {
    "learning_rate": 0.2,
    "max_depth": -1,
    "n_estimators": 200,
    "num_leaves": 100,
    "force_col_wise": True,
}


def main():
    # Load and preprocess the data
    X, y = load_data("datasets/raw/white_wine.csv")
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Convert to contiguous arrays to avoid slicing issues
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    # Define the LightGBM model
    model = lightgbm_model(best_params)

    # Train the model
    train_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
