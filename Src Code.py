# ============================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ============================================

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# 2. Load Dataset
# -----------------------------
# Change path if needed
data_path = "data/house_prices.csv"

df = pd.read_csv(data_path)

print("First 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# 3. Data Cleaning
# -----------------------------

# Remove rows where SalePrice is missing
# (important because SalePrice is target variable)
df = df.dropna(subset=["SalePrice"])

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# -----------------------------
# 4. Encode Categorical Columns
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

print("\nColumns after Encoding:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)


# -----------------------------
# 5. Exploratory Data Analysis
# -----------------------------

# Correlation Heatmap
plt.figure(figsize=(12, 8))

corr = df.corr(numeric_only=True)

sns.heatmap(corr, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("correlation_heatmap.png")

plt.close()


# -----------------------------
# 6. Define Features and Target
# -----------------------------

# Target Variable
y = df["SalePrice"]

# Features
X = df.drop(columns=["SalePrice"])

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


# -----------------------------
# 7. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Size:", X_train.shape[0])
print("Testing Size:", X_test.shape[0])


# -----------------------------
# 8. Train Model
# -----------------------------
# Using Random Forest for better prediction

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")


# -----------------------------
# 9. Model Prediction
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# 10. Model Evaluation
# -----------------------------
r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("-------------------")
print("R² Score:", r2)
print("RMSE:", rmse)


# -----------------------------
# 11. Actual vs Predicted Plot
# -----------------------------
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.close()


# -----------------------------
# 12. Residual Plot
# -----------------------------
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))

plt.scatter(y_test, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Actual Prices")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig("residual_plot.png")

plt.close()


# -----------------------------
# 13. Predict New House Price
# -----------------------------

# Example house input
new_house = pd.DataFrame([{
    "MSSubClass": 60,
    "LotArea": 8450,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "BsmtFinSF2": 0,
    "TotalBsmtSF": 856
}])


# Add missing columns
for col in X.columns:
    if col not in new_house.columns:
        new_house[col] = 0


# Match column order
new_house = new_house[X.columns]


# Predict
predicted_price = model.predict(new_house)[0]

print("\nPredicted Price for New House:")
print(predicted_price)


# -----------------------------
# 14. Save Predictions (Optional)
# -----------------------------
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

results.to_csv("predictions.csv", index=False)

print("\nPredictions saved as predictions.csv")


# -----------------------------
# 15. Final Message
# -----------------------------
print("\nProject Execution Completed Successfully")
print("Generated Files:")
print("- correlation_heatmap.png")
print("- actual_vs_predicted.png")
print("- residual_plot.png")
print("- predictions.csv")