# Data Wrangling 
# Problem Statement: Data Wrangling on Real Estate Market 
# Dataset: "RealEstate_Prices.csv" 
# Description: The dataset contains information about housing prices in a specific real estate 
# market. It includes various attributes such as property characteristics, location, sale prices, 
# and other relevant features. The goal is to perform data wrangling to gain insights into the 
# factors influencing housing prices and prepare the dataset for further analysis or modeling. 
# Tasks to Perform: 
# 1.  Import the "RealEstate_Prices.csv" dataset. Clean column names by removing spaces, 
# special characters, or renaming them for clarity. 
# 2. Handle missing values in the dataset, deciding on an appropriate strategy (e.g., 
# imputation or removal). 
# 3. Perform data merging if additional datasets with relevant information are available 
# (e.g., neighborhood demographics or nearby amenities). 
# 4. Filter and subset the data based on specific criteria, such as a particular time period, 
# property type, or location. 
# 5. Handle categorical variables by encoding them appropriately (e.g., one-hot encoding or 
# label encoding) for further analysis. 
# 6.  Aggregate the data to calculate summary statistics or derived metrics such as average 
# sale prices by neighborhood or property type. 
# 7. Identify and handle outliers or extreme values in the data that may affect the analysis 
# or modeling process. 

# Data Wrangling on Real Estate Market


import pandas as pd
import numpy as np

# 1. Import Dataset
df = pd.read_csv("RealEstate_Prices.csv")

# 2. Clean Column Names
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]", "", regex=True))

print("Columns:", df.columns.tolist())

# 3. Find Price Column
price_col = next(c for c in df.columns if "price" in c)

# Convert Price into Numeric
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

# 4. Handle Missing Values

# Numerical columns - Fill with Median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns - Fill with Mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# 5. Remove Duplicates
df.drop_duplicates(inplace=True)

# 6. Filter Data
filtered = df[df[price_col] > 200000]
print("\nFiltered Data:\n", filtered.head())

# 7. Average Price by Categories
for col in ["zipcode", "bedrooms"]:
    if col in df.columns:
        print("\nAverage Price by", col)
        print(df.groupby(col)[price_col].mean())

# 8. Handle Outliers using IQR
Q1 = df[price_col].quantile(0.25)
Q3 = df[price_col].quantile(0.75)
IQR = Q3 - Q1

df[price_col] = np.clip(
    df[price_col],
    Q1 - 1.5 * IQR,
    Q3 + 1.5 * IQR
)

# 9. Encode Categorical Columns
df = pd.get_dummies(df, dtype=int)

# 10. Export Cleaned Dataset
df.to_csv("RealEstate_Prices_Wrangled.csv", index=False)

print("\nData Wrangling Completed!")
