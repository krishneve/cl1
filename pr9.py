# Data Cleaning and Preparation 
# Problem Statement: Analyzing Customer Churn in a Telecommunications Company 
# Dataset: "Telecom_Customer_Churn.csv" 
# Description: The dataset contains information about customers of a telecommunications 
# company and whether they have churned (i.e., discontinued their services). The dataset 
# includes various attributes of the customers, such as their demographics, usage patterns, and 
# account information. The goal is to perform data cleaning and preparation to gain insights 
# into the factors that contribute to customer churn. 
# Tasks to Perform: 
# 1. Import the "Telecom_Customer_Churn.csv" dataset. 
# 2.  Explore the dataset to understand its structure and content. 
# 3.  Handle missing values in the dataset, deciding on an appropriate strategy. 
# 4. Remove any duplicate records from the dataset. 
# 5.  Check for inconsistent data, such as inconsistent formatting or spelling variations, 
# and standardize it. 
# 6.  Convert columns to the correct data types as needed. 
# 7. Identify and handle outliers in the data. 
# 8. Perform feature engineering, creating new features that may be relevant to 
# predicting customer churn. 
# 9.  Normalize or scale the data if necessary.
# 10. Split the dataset into training and testing sets for further analysis. 
# 11. Export the cleaned dataset for future analysis or modeling.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Import dataset
df = pd.read_csv("Telecom_Customer_Churn.csv")

# 2. Explore dataset
print("First 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# 3. Handle missing values

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Fill missing values in numerical columns
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Fill missing values in categorical columns
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Remove duplicate records
df = df.drop_duplicates()

# 5. Standardize inconsistent text values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# 6. Correct data types
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
df["tenure"] = df["tenure"].astype(int)
df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
df["TotalCharges"] = df["TotalCharges"].astype(float)

# 7. Identify and handle outliers using IQR
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# 8. Feature Engineering
df["AverageCharges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 100],
    labels=["New", "Medium", "Long", "Very Long"]
)

# 11. Export cleaned dataset
df.to_csv("Cleaned_Telecom_Churn.csv", index=False)

# Separate input and output
X = df.drop(["Churn", "customerID"], axis=1, errors="ignore")
y = df["Churn"].map({"No": 0, "Yes": 1})

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 10. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Scale numerical data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nCleaned dataset exported successfully!")
