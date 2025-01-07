# Use the official TensorFlow image as the base
FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose any necessary ports (optional)
# EXPOSE 5000

# Command to run the application
CMD ["python", "wide_deep_mlflow.py"]