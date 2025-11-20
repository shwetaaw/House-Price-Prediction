# house_price_regression.py

# -----------------------------
# 1. Import required libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# For clean plots
plt.style.use("default")


# -----------------------------
# 2. Load the dataset
# -----------------------------
# Make sure your CSV file is in: data/house_prices.csv
# And has columns similar to:
#  Price, Size, Bedrooms, Bathrooms, Floors, YearBuilt, Location

data_path = "data/house_prices.csv"   # change path if needed
df = pd.read_csv(data_path)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())


# -----------------------------
# 3. Basic data cleaning
# -----------------------------

# Rename columns if needed (uncomment & edit according to your file)
# df = df.rename(columns={
#     "price": "Price",
#     "sqft": "Size",
#     "beds": "Bedrooms",
#     "baths": "Bathrooms",
#     "floors": "Floors",
#     "year": "YearBuilt",
#     "location": "Location"
# })

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Simple handling: drop rows with any missing value (for project purpose)
df = df.dropna()

# -----------------------------
# 4. Encode categorical variable (Location)
# -----------------------------
# Location is categorical, so we convert it into dummy/one-hot variables.
# drop_first=True to avoid dummy variable trap.
if "Location" in df.columns:
    df = pd.get_dummies(df, columns=["Location"], drop_first=True)

print("\nColumns after encoding 'Location':")
print(df.columns)


# -----------------------------
# 5. Exploratory Data Analysis (EDA)
# -----------------------------

# (a) Scatter plot: Size vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["Size"], df["Price"])
plt.xlabel("House Size (sq. ft)")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.tight_layout()
plt.savefig("images/size_vs_price.png")  # save plot
plt.close()

# (b) Correlation heatmap
plt.figure(figsize=(9, 7))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()


# -----------------------------
# 6. Define features (X) and target (y)
# -----------------------------

# Target variable
y = df["Price"]

# Feature columns (all except Price)
X = df.drop(columns=["Price"])

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# -----------------------------
# 8. Train Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print("\nIntercept (β0):", model.intercept_)
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nModel Coefficients (β1, β2, ...):")
print(coeff_df)


# -----------------------------
# 9. Model evaluation
# -----------------------------
# Predictions
y_pred = model.predict(X_test)

# R-squared
r2 = r2_score(y_test, y_pred)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print("R-squared (R²):", r2)
print("RMSE:", rmse)

# -----------------------------
# 10. Residual analysis
# -----------------------------
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_test, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("images/residual_plot.png")
plt.close()

print("\nPlots saved in 'images' folder as:")
print("- size_vs_price.png")
print("- correlation_heatmap.png")
print("- residual_plot.png")

# -----------------------------
# 11. Example: Predicting a new house price (optional)
# -----------------------------
# Example input (change values according to your encoded columns)
# Make sure you match the column names EXACTLY as in X.columns

# new_house = pd.DataFrame([{
#     "Size": 1500,
#     "Bedrooms": 3,
#     "Bathrooms": 2,
#     "Floors": 1,
#     "YearBuilt": 2010,
#     # Example for one-hot encoded locations:
#     # "Location_Suburban": 1,
#     # "Location_Urban": 0
# }])

# # Fill any missing dummy columns (if you don't specify all)
# for col in X.columns:
#     if col not in new_house.columns:
#         new_house[col] = 0
# new_house = new_house[X.columns]

# predicted_price = model.predict(new_house)[0]
# print("\nPredicted price for new house:", predicted_price)
