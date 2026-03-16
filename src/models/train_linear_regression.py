import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))

# Load data
data_path = os.path.join(root_dir, "data", "processed", "cleaned_sales_data.csv")
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(columns=['Sales'])
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure output directory exists
output_dir = os.path.join(root_dir, "data", "processed", "model_outputs")
os.makedirs(output_dir, exist_ok=True)

# Train Linear Regression
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)

# Generate predictions
lr_train_pred = lr.predict(X_train)
lr_test_pred = lr.predict(X_test)

# Save predictions
pd.DataFrame(lr_train_pred, columns=['prediction']).to_csv(os.path.join(output_dir, "lr_predictions_train.csv"), index=False)
pd.DataFrame(lr_test_pred, columns=['prediction']).to_csv(os.path.join(output_dir, "lr_predictions_test.csv"), index=False)

# Evaluate model performance
r2 = r2_score(y_test, lr_test_pred)
rmse = mean_squared_error(y_test, lr_test_pred, squared=False)
mae = mean_absolute_error(y_test, lr_test_pred)

print("Linear Regression predictions saved.")
print("\nModel: Linear Regression")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
