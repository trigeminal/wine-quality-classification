import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


def load_data(file):
    data = pd.read_csv(file, delimiter=";")

    # Drop missing values
    data.dropna(inplace=True)

    X = data.drop("quality", axis=1)
    y = data["quality"]

    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure all finite values
    X_train = check_array(X_train, ensure_all_finite=True)
    X_test = check_array(X_test, ensure_all_finite=True)

    return X_train, X_test, y_train, y_test
