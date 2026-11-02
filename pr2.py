
# Uber Fare Prediction using Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("uber.csv")

# 1. Data preprocessing
df = df.dropna()
df = df.drop(columns=["key", "Unnamed: 0"], errors="ignore")

# Select input and output variables
X = df[["pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count"]]

y = df["fare_amount"]

# # 2. Identify outliers using IQR
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(y < lower) | (y > upper)]

print("Number of outliers:", len(outliers))

# Identify outliers using Box Plot

plt.figure(figsize=(6, 4))
plt.boxplot(df["fare_amount"])
plt.title("Box Plot of Uber Fare")
plt.ylabel("Fare Amount")
plt.show()

# Remove outliers
df = df[(y >= lower) & (y <= upper)]

X = df[["pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "passenger_count"]]

y = df["fare_amount"]

plt.figure(figsize=(6, 4))
plt.boxplot(df["fare_amount"])
plt.title("Box Plot of Uber Fare")
plt.ylabel("Fare Amount")
plt.show()

# 3. Check correlation
print("\nCorrelation:")
print(df.corr(numeric_only=True)["fare_amount"])

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Implement regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

# 5. Evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n", name)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

