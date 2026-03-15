# ============================================
# SALES PREDICTION PROJECT - LINEAR REGRESSION MODEL
# File: src/models/linreg_model.py
# Topic: Sales Prediction using Stacking Ensemble
# Role: Linear Regression as Base Learner
# ============================================

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# STEP 2: DEFINE PROJECT PATHS
# ============================================
# Current file -> src/models/linreg_model.py
# Move up 3 levels to reach project root folder: FinalMLproj

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Input cleaned dataset path
data_path = os.path.join(base_path, "data", "processed", "cleaned_sales_data.csv")

# Output folder for model results
output_folder = os.path.join(base_path, "data", "processed", "model_outputs")
os.makedirs(output_folder, exist_ok=True)

plots_folder = os.path.join(output_folder, "linreg_plots")
os.makedirs(plots_folder, exist_ok=True)

# ============================================
# STEP 3: LOAD CLEANED DATASET
# ============================================
print("Loading cleaned dataset...")
print("Looking for file at:", data_path)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found at: {data_path}")

data = pd.read_csv(data_path)

print("Cleaned dataset loaded successfully!")
print("Dataset shape:", data.shape)

# ============================================
# STEP 4: DEFINE FEATURES (X) AND TARGET (y)
# ============================================
print("\nPreparing features and target variable...")

# Target variable
y = data['Sales']

# Feature matrix
X = data.drop('Sales', axis=1)

# Drop redundant column (recommended for Linear Regression)
if 'HasPromoInterval' in X.columns:
    X = X.drop('HasPromoInterval', axis=1)
    print("Dropped redundant column: HasPromoInterval")

# Convert all features to float
X = X.astype(float)

# Remove duplicate columns if any
X = X.loc[:, ~X.columns.duplicated()]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ============================================
# STEP 5: TRAIN-TEST SPLIT (80:20)
# ============================================
print("\nSplitting data into training and testing sets (80:20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ============================================
# STEP 6: CREATE AND TRAIN LINEAR REGRESSION MODEL
# ============================================
print("\nTraining Linear Regression model...")

linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")

# ============================================
# STEP 7: MAKE PREDICTIONS
# ============================================
print("\nMaking predictions on test data...")

y_pred = linreg_model.predict(X_test)

# Sales cannot be negative
y_pred = np.maximum(y_pred, 0)

print("Predictions completed!")

# ============================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================
print("\nEvaluating model performance...")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

train_r2 = linreg_model.score(X_train, y_train)
test_r2 = linreg_model.score(X_test, y_test)

correlation = np.corrcoef(y_test, y_pred)[0, 1]

print("\n================ LINEAR REGRESSION RESULTS ================")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

print("\n================ TRAIN VS TEST PERFORMANCE ================")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R² Score: {test_r2:.4f}")

print(f"\nCorrelation between Actual and Predicted Sales: {correlation:.4f}")

# ============================================
# STEP 9: SAVE PREDICTIONS TO CSV
# ============================================
print("\nSaving actual vs predicted values...")

predictions_df = pd.DataFrame({
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
})

predictions_file = os.path.join(output_folder, "linreg_predictions.csv")
predictions_df.to_csv(predictions_file, index=False)

print("Predictions saved successfully!")
print("Saved at:", predictions_file)

# ============================================
# STEP 10: PLOT 1 - ACTUAL VS PREDICTED SCATTER PLOT
# ============================================
print("\nCreating Actual vs Predicted scatter plot...")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, label="Predicted Points")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linewidth=2,
    label='Ideal Line (Actual = Predicted)'
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Linear Regression: Actual vs Predicted Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "linreg_actual_vs_predicted.png"))
plt.close()

print("Saved: linreg_actual_vs_predicted.png")

# ============================================
# STEP 11: PLOT 2 - RESIDUAL PLOT
# ============================================
print("\nCreating residual plot...")

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "linreg_residual_plot.png"))
plt.close()

print("Saved: linreg_residual_plot.png")

# ============================================
# STEP 12: PLOT 3 - SALES PREDICTION GRAPH (SORTED)
# ============================================
print("\nCreating proper sales prediction graph (sorted actual vs predicted)...")

sorted_df = pd.DataFrame({
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).sort_values(by='Actual_Sales').reset_index(drop=True)

sample_size = 500
if len(sorted_df) > sample_size:
    sample_indices = np.linspace(0, len(sorted_df) - 1, sample_size, dtype=int)
    plot_df = sorted_df.iloc[sample_indices]
else:
    plot_df = sorted_df

plt.figure(figsize=(14, 6))
plt.plot(plot_df['Actual_Sales'].values, label='Actual Sales')
plt.plot(plot_df['Predicted_Sales'].values, label='Predicted Sales')
plt.title("Linear Regression: Sorted Actual vs Predicted Sales")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "linreg_sorted_sales_prediction.png"))
plt.close()

print("Saved: linreg_sorted_sales_prediction.png")

# ============================================
# STEP 13: PLOT 4 - FIRST 100 ACTUAL VS PREDICTED VALUES
# ============================================
print("\nCreating comparison line plot for first 100 test samples...")

comparison_df = pd.DataFrame({
    'Actual': y_test.values[:100],
    'Predicted': y_pred[:100]
})

plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Actual'].values, label='Actual Sales')
plt.plot(comparison_df['Predicted'].values, label='Predicted Sales')
plt.title("Linear Regression: Actual vs Predicted Sales (First 100 Test Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "linreg_first100_comparison.png"))
plt.close()

print("Saved: linreg_first100_comparison.png")

# ============================================
# STEP 14: PLOT 5 - ERROR METRICS BAR CHART
# ============================================
print("\nCreating error metrics bar chart...")

metric_names = ['MAE', 'RMSE']
metric_values = [mae, rmse]

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, metric_values)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.2f}',
        ha='center',
        va='bottom'
    )

plt.title("Linear Regression Error Metrics")
plt.ylabel("Error Value")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "linreg_error_metrics_bar_chart.png"))
plt.close()

print("Saved: linreg_error_metrics_bar_chart.png")

# ============================================
# STEP 15: DISPLAY SAMPLE PREDICTIONS
# ============================================
print("\n================ SAMPLE PREDICTIONS ================")
print(predictions_df.head(10))

# ============================================
# STEP 16: FEATURE COEFFICIENTS
# ============================================
print("\nFinding top feature coefficients...")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linreg_model.coef_
})

coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False).reset_index(drop=True)

print("\n================ TOP 10 IMPORTANT FEATURES (LINEAR REGRESSION) ================")
print(coefficients[['Feature', 'Coefficient']].head(10).to_string(index=False))

coeff_file = os.path.join(output_folder, "linreg_feature_coefficients.csv")
coefficients.to_csv(coeff_file, index=False)

print("\nFeature coefficients saved successfully!")
print("Saved at:", coeff_file)

# ============================================
# STEP 17: SAVE MODEL METRICS TO CSV
# ============================================
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2_Score', 'Train_R2', 'Test_R2', 'Correlation'],
    'Value': [mae, rmse, r2, train_r2, test_r2, correlation]
})

metrics_file = os.path.join(output_folder, "linreg_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)

print("\nModel metrics saved successfully!")
print("Saved at:", metrics_file)

# ============================================
# STEP 18: FINAL CONCLUSION
# ============================================
print("\n================ FINAL CONCLUSION ================")
print("Linear Regression was used as a baseline model for sales prediction.")
print("It captures the basic linear relationship between input features and sales.")
print("However, because sales data often has non-linear patterns, the model may underfit.")
print("This makes Linear Regression a useful base learner in the Stacking Ensemble project,")
print("where stronger models like Random Forest and SVR can improve overall performance.")

# ============================================
# STEP 19: FINAL MESSAGE
# ============================================
print("\n==============================================")
print("LINEAR REGRESSION MODEL COMPLETED SUCCESSFULLY!")
print("Role in project: Base learner for Stacking Ensemble")
print("Input file used:")
print("data/processed/cleaned_sales_data.csv")
print("\nOutputs saved in:")
print("data/processed/model_outputs/")
print("==============================================")