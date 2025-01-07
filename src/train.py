import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlflow.models.signature import infer_signature


def train_model(model, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_params(model.get_params())

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Log the metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        # Define input example and signature
        input_example = X_train[
            :1
        ]  # Use the first row of training data as an input example
        signature = infer_signature(input_example, model.predict(input_example))

        # Log the model with signature and input example
        mlflow.lightgbm.log_model(
            model,
            "lightgbm_model",
            signature=signature,
            input_example=input_example,
        )

        # Print the metrics
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
