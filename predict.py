from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# Define the dataset folder
dataset_folder = "../dataset/"

print("Loading dataset...")
# Load testing data
X_test = pd.read_csv(f"{dataset_folder}X_test.csv")
y_test = pd.read_csv(f"{dataset_folder}y_test.csv")
print("Dataset loaded.")

# Ensure y_test is a Series (1D) rather than a DataFrame
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]

# Load the PyCaret-tuned model
model = load_model('local/tuned_regression_model_1')

# Make predictions on the test set
predictions = predict_model(model, data=X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions['prediction_label'])
print(f"Mean Squared Error: {mse}")

# Load the saved model and scalers
model_path = "local/random_forest_model.joblib"  # Replace with your saved model's filename
scaler_path = "local/scaler.pkl"

print("Loading model and scalers...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("Model and scalers loaded.")

# Standardize test features
X_test_scaled = scaler.transform(X_test)

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test_scaled)
predictions = pd.DataFrame({'y_hat': predictions})
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
