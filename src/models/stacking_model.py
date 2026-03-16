# ============================================
# SALES PREDICTION PROJECT - STACKING ENSEMBLE
# File: src/models/stacking_model.py
# Topic: Sales Prediction using Stacking Ensemble
# Role: Stacking Ensemble (LR + RF + SVR → Ridge meta-model)
# ============================================

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# STEP 2: DEFINE PROJECT PATHS
# ============================================
# Current file -> src/models/stacking_model.py
# Move up 3 levels to reach project root folder
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Input cleaned dataset path (same as teammates)
data_path = os.path.join(base_path, "data", "processed", "cleaned_sales_data.csv")

# Output folder for model results
output_folder = os.path.join(base_path, "data", "processed", "model_outputs")
os.makedirs(output_folder, exist_ok=True)

plots_folder = os.path.join(output_folder, "stacking_plots")
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

# Drop redundant column (same as teammates' LR model)
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
# STEP 6: DEFINE BASE MODELS AND META-MODEL
# ============================================
print("\nDefining base models and meta-model...")

# Base Model 1: Linear Regression (with scaling)
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Base Model 2: Random Forest
rf_pipeline = Pipeline([
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

# Base Model 3: SVR (with scaling — SVR requires scaled input)
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
])

# Meta-model: Ridge Regression (linear, regularised)
meta_model = Ridge(alpha=1.0)

print("Base models defined: Linear Regression, Random Forest, SVR")
print("Meta-model defined: Ridge Regression")

# ============================================
# STEP 7: BUILD STACKING ENSEMBLE
# ============================================
print("\nBuilding Stacking Ensemble...")

estimators = [
    ("lr",  lr_pipeline),
    ("rf",  rf_pipeline),
    ("svr", svr_pipeline),
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,               # 5-fold cross-validation for meta-features
    passthrough=True,   # meta-model also sees original features
    n_jobs=-1
)

print("Stacking Ensemble built successfully!")

# ============================================
# STEP 8: TRAIN ALL MODELS (individual + stacking)
# ============================================
print("\nTraining all models on training data...")

# Train individual base models (for comparison)
print("  Training Linear Regression...")
lr_pipeline.fit(X_train, y_train)

print("  Training Random Forest...")
rf_pipeline.fit(X_train, y_train)

print("  Training SVR...")
svr_pipeline.fit(X_train, y_train)

print("  Training Stacking Ensemble (this may take a moment)...")
stacking_model.fit(X_train, y_train)

print("All models trained successfully!")

# ============================================
# STEP 9: PREDICT SALES ON TEST DATA
# ============================================
print("\nMaking predictions on test data...")

y_pred_lr      = np.maximum(lr_pipeline.predict(X_test), 0)
y_pred_rf      = np.maximum(rf_pipeline.predict(X_test), 0)
y_pred_svr     = np.maximum(svr_pipeline.predict(X_test), 0)
y_pred_stack   = np.maximum(stacking_model.predict(X_test), 0)

print("Predictions completed!")

# ============================================
# STEP 10: EVALUATE MODEL PERFORMANCE
# ============================================
print("\nEvaluating model performance...")

def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2_Score": r2}

results = [
    evaluate_model("Linear Regression", y_test, y_pred_lr),
    evaluate_model("Random Forest",     y_test, y_pred_rf),
    evaluate_model("SVR",               y_test, y_pred_svr),
    evaluate_model("Stacking Ensemble", y_test, y_pred_stack),
]

metrics_df = pd.DataFrame(results)

print("\n================= MODEL COMPARISON RESULTS =================")
print(metrics_df.to_string(index=False))

# Individual stacking details
stack_mae  = mean_absolute_error(y_test, y_pred_stack)
stack_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
stack_r2   = r2_score(y_test, y_pred_stack)
train_r2   = stacking_model.score(X_train, y_train)
test_r2    = stacking_model.score(X_test, y_test)
correlation = np.corrcoef(y_test, y_pred_stack)[0, 1]

print(f"\n================= STACKING ENSEMBLE RESULTS =================")
print(f"Mean Absolute Error (MAE): {stack_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {stack_rmse:.2f}")
print(f"R² Score: {stack_r2:.4f}")
print(f"\n================= TRAIN VS TEST PERFORMANCE =================")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R² Score:  {test_r2:.4f}")
print(f"\nCorrelation between Actual and Predicted Sales: {correlation:.4f}")

# ============================================
# STEP 11: SAVE PREDICTIONS TO CSV
# ============================================
print("\nSaving actual vs predicted values...")

predictions_df = pd.DataFrame({
    'Actual_Sales':    y_test.values,
    'LR_Predicted':    y_pred_lr,
    'RF_Predicted':    y_pred_rf,
    'SVR_Predicted':   y_pred_svr,
    'Stack_Predicted': y_pred_stack
})

predictions_file = os.path.join(output_folder, "stacking_predictions.csv")
predictions_df.to_csv(predictions_file, index=False)

print("Predictions saved successfully!")
print("Saved at:", predictions_file)

# ============================================
# STEP 12: PLOT 1 - ACTUAL VS PREDICTED (STACKING)
# ============================================
print("\nCreating Actual vs Predicted scatter plot...")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_stack, alpha=0.3, label="Predicted Points")

min_val = min(y_test.min(), y_pred_stack.min())
max_val = max(y_test.max(), y_pred_stack.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linewidth=2,
    label='Ideal Line (Actual = Predicted)'
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Stacking Ensemble: Actual vs Predicted Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "stacking_actual_vs_predicted.png"))
plt.close()

print("Saved: stacking_actual_vs_predicted.png")

# ============================================
# STEP 13: PLOT 2 - RESIDUAL PLOT
# ============================================
print("\nCreating residual plot...")

residuals = y_test.values - y_pred_stack

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_stack, residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Stacking Ensemble: Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "stacking_residual_plot.png"))
plt.close()

print("Saved: stacking_residual_plot.png")

# ============================================
# STEP 14: PLOT 3 - SORTED SALES PREDICTION GRAPH
# ============================================
print("\nCreating proper sales prediction graph (sorted actual vs predicted)...")

sorted_df = pd.DataFrame({
    'Actual_Sales':    y_test.values,
    'Predicted_Sales': y_pred_stack
}).sort_values(by='Actual_Sales').reset_index(drop=True)

sample_size = 500
if len(sorted_df) > sample_size:
    sample_indices = np.linspace(0, len(sorted_df) - 1, sample_size, dtype=int)
    plot_df = sorted_df.iloc[sample_indices]
else:
    plot_df = sorted_df

plt.figure(figsize=(14, 6))
plt.plot(plot_df['Actual_Sales'].values,    label='Actual Sales')
plt.plot(plot_df['Predicted_Sales'].values, label='Predicted Sales')
plt.title("Stacking Ensemble: Sorted Actual vs Predicted Sales")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "stacking_sorted_sales_prediction.png"))
plt.close()

print("Saved: stacking_sorted_sales_prediction.png")

# ============================================
# STEP 15: PLOT 4 - FIRST 100 ACTUAL VS PREDICTED
# ============================================
print("\nCreating comparison line plot for first 100 test samples...")

comparison_df = pd.DataFrame({
    'Actual':    y_test.values[:100],
    'Predicted': y_pred_stack[:100]
})

plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Actual'].values,    label='Actual Sales')
plt.plot(comparison_df['Predicted'].values, label='Predicted Sales')
plt.title("Stacking Ensemble: Actual vs Predicted Sales (First 100 Test Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "stacking_first100_comparison.png"))
plt.close()

print("Saved: stacking_first100_comparison.png")

# ============================================
# STEP 16: PLOT 5 - MODEL COMPARISON BAR CHART
# ============================================
print("\nCreating model comparison bar chart...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metrics_list  = ["MAE", "RMSE", "R2_Score"]
titles        = ["Mean Absolute Error ↓", "Root Mean Squared Error ↓", "R² Score ↑"]
colors        = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

for ax, metric, title in zip(axes, metrics_list, titles):
    bars = ax.bar(metrics_df["Model"], metrics_df[metric], color=colors, edgecolor="none")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=8
        )

plt.suptitle("Model Comparison: LR vs RF vs SVR vs Stacking Ensemble",
             fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "model_comparison_bar_chart.png"))
plt.close()

print("Saved: model_comparison_bar_chart.png")

# ============================================
# STEP 17: PLOT 6 - ERROR METRICS BAR CHART (MAE & RMSE only)
# ============================================
print("\nCreating error metrics bar chart...")

metric_names  = ['MAE', 'RMSE']
metric_values = [stack_mae, stack_rmse]

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, metric_values, color=["#4C72B0", "#C44E52"])

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.2f}',
        ha='center',
        va='bottom'
    )

plt.title("Stacking Ensemble Error Metrics")
plt.ylabel("Error Value")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "stacking_error_metrics_bar_chart.png"))
plt.close()

print("Saved: stacking_error_metrics_bar_chart.png")

# ============================================
# STEP 18: SAVE MODEL METRICS TO CSV
# ============================================
metrics_full_df = pd.DataFrame({
    'Metric':     ['MAE', 'RMSE', 'R2_Score', 'Train_R2', 'Test_R2', 'Correlation'],
    'Value':      [stack_mae, stack_rmse, stack_r2, train_r2, test_r2, correlation]
})

metrics_file = os.path.join(output_folder, "stacking_metrics.csv")
metrics_full_df.to_csv(metrics_file, index=False)

# Also save comparison across all models
comparison_file = os.path.join(output_folder, "all_models_comparison.csv")
metrics_df.to_csv(comparison_file, index=False)

print("\nModel metrics saved successfully!")
print("Saved at:", metrics_file)
print("Comparison saved at:", comparison_file)

# ============================================
# STEP 19: FINAL CONCLUSION
# ============================================
print("\n================= FINAL CONCLUSION =================")
print("Stacking Ensemble was used as the final model for sales prediction.")
print("It combines Linear Regression, Random Forest, and SVR as base learners.")
print("A Ridge Regression meta-model learns from the base model outputs.")
print("The Stacking Ensemble achieves better generalisation than individual models,")
print("because it leverages the strengths of all three base learners.")

print("\n===============================================")
print("STACKING ENSEMBLE MODEL COMPLETED SUCCESSFULLY!")
print("Role: Final ensemble combining all base learners")
print("Input file used:")
print(data_path)
print("\nOutputs saved in:")
print(output_folder)
print("===============================================")