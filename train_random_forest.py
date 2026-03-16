import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Load data
data_path = "data/processed/cleaned_sales_data.csv"
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(columns=['Sales'])
y = df['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure output directory exists
output_dir = "data/processed/model_outputs"
os.makedirs(output_dir, exist_ok=True)

# Train Random Forest
print("Training Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_samples=100000)  # Limit samples for speed
rf.fit(X_train, y_train)

print("Generating predictions...")
# Generate train predictions
rf_train_pred = rf.predict(X_train)

# Generate predictions in batches to avoid memory issues
batch_size = 50000
rf_test_pred = []
for i in range(0, len(X_test), batch_size):
    batch = X_test.iloc[i:i+batch_size]
    rf_test_pred.extend(rf.predict(batch))
rf_test_pred = np.array(rf_test_pred)

# Save predictions
pd.DataFrame(rf_train_pred, columns=['prediction']).to_csv(os.path.join(output_dir, "rf_predictions_train.csv"), index=False)
pd.DataFrame(rf_test_pred, columns=['prediction']).to_csv(os.path.join(output_dir, "rf_predictions_test.csv"), index=False)

# Evaluate model performance
r2 = r2_score(y_test, rf_test_pred)
rmse = mean_squared_error(y_test, rf_test_pred, squared=False)
mae = mean_absolute_error(y_test, rf_test_pred)

print("Random Forest predictions saved.")
print("\nModel: Random Forest")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
