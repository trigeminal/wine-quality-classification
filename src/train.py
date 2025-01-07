import mlflow
import mlflow.tensorflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    with mlflow.start_run():
        mlflow.log_param("model_type", "Wide & Deep Network")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # Train the model
        history = model.fit(
            [X_train, X_train],
            y_train,
            validation_data=([X_test, X_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        # Evaluate the model
        y_pred = model.predict([X_test, X_test])

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Log the metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        # Define the input signature
        input_example = {"wide_input": X_train[:1], "deep_input": X_train[:1]}
        signature = infer_signature(
            input_example,
            model.predict([input_example["wide_input"], input_example["deep_input"]]),
        )

        # Log the model
        mlflow.tensorflow.log_model(
            model, "wide_deep_model", signature=signature, input_example=input_example
        )

        # Print the metrics
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
