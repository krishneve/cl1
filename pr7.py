# Part II: Data Modeling and Visualization 
# 7 Data Loading, Storage and File Formats 
# Problem Statement: Analyzing Sales Data from Multiple File Formats 
# Dataset: Sales data in multiple file formats (e.g., CSV, Excel, JSON) 
# Description: The goal is to load and analyze sales data from different file formats, including 
# CSV, Excel, and JSON, and perform data cleaning, transformation, and analysis on the 
# dataset. 
  
# Tasks to Perform: 
#  Obtain sales data files in various formats, such as CSV, Excel, and JSON. 
# 1. Load the sales data from each file format into the appropriate data structures or 
# dataframes. 
# 2. Explore the structure and content of the loaded data, identifying any inconsistencies, 
# missing values, or data quality issues. 
# 3. Perform data cleaning operations, such as handling missing values, removing 
# duplicates, or correcting inconsistencies. 
# 4. Convert the data into a unified format, such as a common dataframe or data structure, 
# to enable seamless analysis.

# 5. Perform data transformation tasks, such as merging multiple datasets, splitting 
# columns, or deriving new variables. 
# 6. Analyze the sales data by performing descriptive statistics, aggregating data by 
# specific variables, or calculating metrics such as total sales, average order value, or 
# product category distribution. 
# 7. Create visualizations, such as bar plots, pie charts, or box plots, to represent the sales 
# data and gain insights into sales trends, customer behavior, or product performance. 


import pandas as pd
import matplotlib.pyplot as plt

# Load dataset with correct encoding
df = pd.read_csv("Superstore.csv", encoding="latin1")

print(df.head())
print(df.info())

# Convert into Excel and JSON
df.to_excel("sales.xlsx", index=False)
df.to_json("sales.json", orient="records")

# Load different formats
csv = pd.read_csv("Superstore.csv", encoding="latin1")
excel = pd.read_excel("sales.xlsx")
json = pd.read_json("sales.json")

# Combine and clean
df = pd.concat([csv, excel, json], ignore_index=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("\nDescriptive Statistics:")
print(df.describe())

print("\nTotal Sales:", df["Sales"].sum())

# Sales by Category
sales = df.groupby("Category")["Sales"].sum()
print(sales)

# Bar Plot
sales.plot(kind="bar", title="Sales by Category")
plt.ylabel("Total Sales")
plt.show()

# Pie Chart
sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("Category-wise Sales")
plt.ylabel("")
plt.show()