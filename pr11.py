# Data Visualization using matplotlib 
# Problem Statement: Analyzing Air Quality Index (AQI) Trends in a City  
# Dataset: "City_Air_Quality.csv" 
# Description: The dataset contains information about air quality measurements in a specific 
# city over a period of time. It includes attributes such as date, time, pollutant levels (e.g., PM2.5, 
# PM10, CO), and the Air Quality Index (AQI) values. The goal is to use the matplotlib library 
# to create visualizations that effectively represent the AQI trends and patterns for different 
# pollutants in the city.  
# Tasks to Perform:  
# 1.  Import the "City_Air_Quality.csv" dataset. 
# 2.  Explore the dataset to understand its structure and content. 
# 3. Identify the relevant variables for visualizing AQI trends, such as date, pollutant levels, 
# and AQI values. 
# 4. Create line plots or time series plots to visualize the overall AQI trend over time. 
# 5. Plot individual pollutant levels (e.g., PM2.5, PM10, CO) on separate line plots to 
# visualize their trends over time. 
# 6. Use bar plots or stacked bar plots to compare the AQI values across different dates or 
# time periods. 
# 7. Create box plots or violin plots to analyze the distribution of AQI values for different 
# pollutant categories. 
# 8. Use scatter plots or bubble charts to explore the relationship between AQI values and 
# pollutant levels. 
# 9. Customize the visualizations by adding labels, titles, legends, and appropriate color 
# schemes. 

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("City_Air_Quality.csv")

# 2. Explore data
print(df.head())
print(df.info())
print(df.isnull().sum())

# 3. Data cleaning
df = df.drop_duplicates()
df = df.dropna()

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# 4. AQI Trend - Line Plot
plt.plot(df["Date"], df["AQI"], label="AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.title("AQI Trend")
plt.legend()
plt.show()

# 5. Pollutant Trends
df.plot(x="Date", y=["PM2.5", "PM10", "CO"])
plt.title("Pollutant Trends")
plt.show()

# 6. Bar Plot
df.plot(x="Date", y="AQI", kind="bar", title="Daily AQI")
plt.show()

# 7. Box Plot
df.boxplot(column="AQI")
plt.title("AQI Distribution")
plt.show()

# 8. Scatter Plot
plt.scatter(df["PM2.5"], df["AQI"])
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.title("PM2.5 vs AQI")
plt.show()
