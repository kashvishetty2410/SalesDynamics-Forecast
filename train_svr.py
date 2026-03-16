import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import time

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

# Subsample for SVR training to make it feasible
sample_size = 25000  # Try more samples with linear kernel
if len(X_train) > sample_size:
    X_train_svr = X_train.sample(n=sample_size, random_state=42)
    y_train_svr = y_train.loc[X_train_svr.index]
else:
    X_train_svr = X_train
    y_train_svr = y_train

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_svr_scaled = scaler.fit_transform(X_train_svr)
X_test_scaled = scaler.transform(X_test)

# Train SVR with better parameters
print(f"Training SVR on {len(X_train_svr)} samples...")
start_time = time.time()
svr = SVR(kernel="linear", C=100)  # Try linear kernel with moderate C
svr.fit(X_train_svr_scaled, y_train_svr)
print(f"SVR training completed in {time.time() - start_time:.2f} seconds")

# Generate predictions for the full train and test sets
print("Generating predictions...")
# Scale full training data for prediction
X_train_scaled = scaler.transform(X_train)
svr_train_pred = svr.predict(X_train_scaled)
svr_test_pred = svr.predict(X_test_scaled)

# Save predictions
pd.DataFrame(svr_train_pred, columns=['prediction']).to_csv(os.path.join(output_dir, "svr_predictions_train.csv"), index=False)
pd.DataFrame(svr_test_pred, columns=['prediction']).to_csv(os.path.join(output_dir, "svr_predictions_test.csv"), index=False)

# Evaluate model performance
r2 = r2_score(y_test, svr_test_pred)
rmse = mean_squared_error(y_test, svr_test_pred, squared=False)
mae = mean_absolute_error(y_test, svr_test_pred)

print("SVR predictions saved.")
print("\nModel: Support Vector Regression")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
