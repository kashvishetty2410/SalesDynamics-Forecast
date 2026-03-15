# ============================================
# SALES PREDICTION PROJECT - PREPROCESSING + EDA
# File: src/preprocessing/preprocess_eda.py
# ============================================

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display all columns
pd.set_option('display.max_columns', None)

# ============================================
# STEP 2: DEFINE PROJECT PATHS
# ============================================
# Current file -> src/preprocessing/preprocess_eda.py
# Move up 2 levels to reach project root folder: FinalMLproj

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_folder = os.path.join(base_path, "data")
train_path = os.path.join(data_folder, "train.csv")
store_path = os.path.join(data_folder, "store.csv")

# Output folder for cleaned data and plots
output_folder = os.path.join(data_folder, "processed")
os.makedirs(output_folder, exist_ok=True)

plots_folder = os.path.join(output_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

# ============================================
# STEP 3: LOAD DATASETS
# ============================================
print("Loading datasets...")

# FIX 1: Read StateHoliday as string to avoid mixed type warning
train = pd.read_csv(train_path, dtype={'StateHoliday': str}, low_memory=False)
store = pd.read_csv(store_path)

print("Train dataset loaded successfully!")
print("Store dataset loaded successfully!")

# ============================================
# STEP 4: DISPLAY FIRST FEW ROWS
# ============================================
print("\n================ TRAIN DATASET PREVIEW ================")
print(train.head())

print("\n================ STORE DATASET PREVIEW ================")
print(store.head())

# ============================================
# STEP 5: CHECK SHAPE OF BOTH DATASETS
# ============================================
print("\n================ DATASET SHAPES ================")
print("Train dataset shape:", train.shape)
print("Store dataset shape:", store.shape)

# ============================================
# STEP 6: CHECK COLUMN NAMES
# ============================================
print("\n================ TRAIN COLUMNS ================")
print(train.columns.tolist())

print("\n================ STORE COLUMNS ================")
print(store.columns.tolist())

# ============================================
# STEP 7: MERGE BOTH DATASETS USING 'Store'
# ============================================
print("\nMerging train and store datasets on 'Store' column...")

data = pd.merge(train, store, on='Store', how='left')

print("Datasets merged successfully!")

print("\n================ MERGED DATASET PREVIEW ================")
print(data.head())

print("\nMerged dataset shape:", data.shape)

# ============================================
# STEP 8: BASIC INFORMATION ABOUT MERGED DATA
# ============================================
print("\n================ MERGED DATASET INFO ================")
data.info()

# ============================================
# STEP 9: CHECK DATA TYPES
# ============================================
print("\n================ DATA TYPES ================")
print(data.dtypes)

# ============================================
# STEP 10: CHECK MISSING VALUES
# ============================================
print("\n================ MISSING VALUES ================")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# ============================================
# STEP 11: CHECK DUPLICATE ROWS
# ============================================
print("\n================ DUPLICATE ROWS ================")
duplicate_count = data.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)

# ============================================
# STEP 12: CONVERT DATE COLUMN TO DATETIME
# ============================================
print("\nConverting 'Date' column to datetime format...")

data['Date'] = pd.to_datetime(data['Date'])

print("Date column converted successfully!")
print(data['Date'].head())

# ============================================
# STEP 13: SORT DATA BY DATE (IMPORTANT FOR TIME SERIES)
# ============================================
print("\nSorting data by Date for time series analysis...")

# FIX 3: Keep this sort before feature creation and before dropping Date
data = data.sort_values(by='Date').reset_index(drop=True)

print("Data sorted successfully!")

# ============================================
# STEP 14: EXTRACT DATE-BASED FEATURES
# ============================================
print("\nCreating time-based features from Date column...")

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['WeekOfYear'] = data['Date'].dt.isocalendar().week.astype(int)
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x in [6, 7] else 0)

print("Time-based features created successfully!")

# ============================================
# STEP 15: DESCRIPTIVE STATISTICS
# ============================================
print("\n================ NUMERICAL SUMMARY ================")
print(data.describe())

print("\n================ CATEGORICAL SUMMARY ================")
print(data.describe(include='object'))

# ============================================
# STEP 16: CHECK UNIQUE VALUES IN IMPORTANT CATEGORICAL COLUMNS
# ============================================
print("\n================ UNIQUE VALUES IN CATEGORICAL COLUMNS ================")

categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']

for col in categorical_cols:
    if col in data.columns:
        print(f"\nUnique values in {col}:")
        print(data[col].unique())

# ============================================
# STEP 17: HANDLE MISSING VALUES PROPERLY
# ============================================
print("\nHandling missing values...")

# Competition distance -> numerical, use median
if 'CompetitionDistance' in data.columns:
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].median())

# Competition open month/year -> missing means unknown, fill with 0
if 'CompetitionOpenSinceMonth' in data.columns:
    data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].fillna(0)

if 'CompetitionOpenSinceYear' in data.columns:
    data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].fillna(0)

# Promo2SinceWeek/Year -> fill with 0
if 'Promo2SinceWeek' in data.columns:
    data['Promo2SinceWeek'] = data['Promo2SinceWeek'].fillna(0)

if 'Promo2SinceYear' in data.columns:
    data['Promo2SinceYear'] = data['Promo2SinceYear'].fillna(0)

# PromoInterval -> missing means no promo interval
if 'PromoInterval' in data.columns:
    data['PromoInterval'] = data['PromoInterval'].fillna('NoPromo')

print("Missing values handled successfully!")

# ============================================
# STEP 18: VERIFY MISSING VALUES AFTER CLEANING
# ============================================
print("\n================ MISSING VALUES AFTER CLEANING ================")
print(data.isnull().sum()[data.isnull().sum() > 0])

# ============================================
# STEP 19: FILTER CLOSED STORES (OPTIONAL BUT IMPORTANT)
# ============================================
# For sales prediction, rows where Open = 0 usually have Sales = 0
# Many projects remove them to improve model performance

print("\nFiltering only open stores (Open = 1)...")

if 'Open' in data.columns:
    data = data[data['Open'] == 1]

print("Rows after filtering open stores:", data.shape)

# ============================================
# STEP 20: REMOVE ZERO SALES ROWS (IMPORTANT FOR REGRESSION)
# ============================================
print("\nRemoving rows where Sales = 0...")

data = data[data['Sales'] > 0]

print("Rows after removing zero sales:", data.shape)

# ============================================
# STEP 21: SIMPLE EDA - SALES OVER TIME (TIME SERIES TREND)
# ============================================
print("\nCreating sales trend plot...")

daily_sales = data.groupby('Date')['Sales'].sum()

plt.figure(figsize=(12, 6))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "daily_sales_trend.png"))
plt.close()

print("Saved: daily_sales_trend.png")

# ============================================
# STEP 22: SIMPLE EDA - MONTHLY SALES (SEASONAL TREND)
# ============================================
print("\nCreating monthly seasonal sales plot...")

monthly_sales = data.groupby('Month')['Sales'].mean()

plt.figure(figsize=(10, 5))
plt.bar(monthly_sales.index, monthly_sales.values)
plt.title("Average Sales by Month (Seasonality)")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "monthly_sales_seasonality.png"))
plt.close()

print("Saved: monthly_sales_seasonality.png")

# ============================================
# STEP 23: SIMPLE EDA - SALES BY DAY OF WEEK
# ============================================
print("\nCreating sales by day of week plot...")

dow_sales = data.groupby('DayOfWeek')['Sales'].mean()

plt.figure(figsize=(8, 5))
plt.bar(dow_sales.index, dow_sales.values)
plt.title("Average Sales by Day Of Week")
plt.xlabel("Day Of Week")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "sales_by_dayofweek.png"))
plt.close()

print("Saved: sales_by_dayofweek.png")

# ============================================
# STEP 24: SIMPLE EDA - PROMO EFFECT ON SALES
# ============================================
print("\nCreating promo vs non-promo sales plot...")

promo_sales = data.groupby('Promo')['Sales'].mean()

plt.figure(figsize=(6, 4))
plt.bar(promo_sales.index.astype(str), promo_sales.values)
plt.title("Average Sales: Promo vs No Promo")
plt.xlabel("Promo (0 = No, 1 = Yes)")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "promo_effect_on_sales.png"))
plt.close()

print("Saved: promo_effect_on_sales.png")

# ============================================
# STEP 25: SIMPLE EDA - COMPETITION DISTANCE VS SALES
# ============================================
print("\nCreating competition distance vs sales plot...")

plt.figure(figsize=(8, 5))
plt.scatter(data['CompetitionDistance'], data['Sales'], alpha=0.3)
plt.title("Competition Distance vs Sales")
plt.xlabel("Competition Distance")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "competition_distance_vs_sales.png"))
plt.close()

print("Saved: competition_distance_vs_sales.png")

# ============================================
# STEP 26: FEATURE ENGINEERING FOR PROMO INTERVAL
# ============================================
print("\nCreating promo interval feature...")

# Whether store has Promo2 active
data['HasPromoInterval'] = data['PromoInterval'].apply(lambda x: 0 if x == 'NoPromo' else 1)

print("Promo interval feature created successfully!")

# ============================================
# STEP 27: ENCODE CATEGORICAL COLUMNS
# ============================================
print("\nEncoding categorical columns...")

# FIX 1 continued: Convert StateHoliday to string before encoding
data['StateHoliday'] = data['StateHoliday'].astype(str)

# One-hot encode categorical columns
categorical_features = ['StoreType', 'Assortment', 'StateHoliday']

# PromoInterval is text-heavy, we already created HasPromoInterval
# so we can drop original PromoInterval later

data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

print("Categorical encoding completed!")

# ============================================
# STEP 28: DROP UNUSED / HIGHLY TEXTUAL COLUMNS
# ============================================
print("\nDropping unnecessary columns...")

# FIX 2: Drop Customers to avoid data leakage
columns_to_drop = ['Date', 'PromoInterval', 'Customers']

for col in columns_to_drop:
    if col in data.columns:
        data.drop(columns=col, inplace=True)

print("Unnecessary columns dropped!")

# ============================================
# STEP 29: FINAL CHECK
# ============================================
print("\n================ FINAL DATASET INFO ================")
data.info()

print("\n================ FINAL DATASET PREVIEW ================")
print(data.head())

print("\nFinal dataset shape:", data.shape)

# ============================================
# STEP 30: SAVE CLEANED DATASET
# ============================================
cleaned_file_path = os.path.join(output_folder, "cleaned_sales_data.csv")

data.to_csv(cleaned_file_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Saved at:", cleaned_file_path)

# ============================================
# STEP 31: PRINT FINAL MESSAGE
# ============================================
print("\n==============================================")
print("PREPROCESSING + EDA COMPLETED SUCCESSFULLY!")
print("Use this file for all models:")
print("data/processed/cleaned_sales_data.csv")
print("Plots saved in:")
print("data/processed/plots/")
print("==============================================")