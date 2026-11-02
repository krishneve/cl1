# Data Aggregation 
# Problem Statement: Analyzing Sales Performance by Region in a Retail Company 
# Dataset: "Retail_Sales_Data.csv" 
# Description: The dataset contains information about sales transactions in a retail company. It 
# includes attributes such as transaction date, product category, quantity sold, and sales 
# amount. The goal is to perform data aggregation to analyze the sales performance by region 
# and identify the top-performing regions. 
# Tasks to Perform: 
# 1. Import the "Retail_Sales_Data.csv" dataset. 
# 2. Explore the dataset to understand its structure and content. 
# 3. Identify the relevant variables for aggregating sales data, such as region, sales 
# amount, and product category. 
# 4. Group the sales data by region and calculate the total sales amount for each region. 
# 5. Create bar plots or pie charts to visualize the sales distribution by region. 
# 6. Identify the top-performing regions based on the highest sales amount. 
# 7. Group the sales data by region and product category to calculate the total sales 
# amount for each combination. 
# 8. Create stacked bar plots or grouped bar plots to compare the sales amounts across 
# different regions and product categories.
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("Retail_Sales_Data1.csv", encoding="latin1")

# 2. Explore dataset
print(df.head())
print(df.info())
print(df.describe())

# 3. Select required columns
df = df[["Region", "Category", "Sales"]]

# 4. Total sales by region
region_sales = df.groupby("Region")["Sales"].sum()
print("\nSales by Region:")
print(region_sales)

# 5. Bar plot
region_sales.plot(kind="bar", title="Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()

# 6. Top performing region
print("\nTop Region:", region_sales.idxmax())
print("Highest Sales:", region_sales.max())

# 7. Sales by region and category
category_sales = df.groupby(
    ["Region", "Category"]
)["Sales"].sum().unstack()

print("\nRegion and Category Sales:")
print(category_sales)

# 8. Stacked bar plot
category_sales.plot(kind="bar", stacked=True)
plt.title("Region-wise Category Sales")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()