
import requests
import pandas as pd
import matplotlib.pyplot as plt

# 1. API Request
api_key = "2962dc3cde1c1558bf8bb89eab13d534"
url = "https://api.openweathermap.org/data/2.5/forecast"

params = {"q": "Nashik", "appid": api_key, "units": "metric"}
data = requests.get(url, params=params).json()

if "list" in data:

    # 2. Extract Weather Data
    df = pd.DataFrame([{
        "Date": x["dt_txt"],
        "Temp": x["main"]["temp"],
        "Humidity": x["main"]["humidity"],
        "Wind": x["wind"]["speed"],
        "Rain": x.get("rain", {}).get("3h", 0)
    } for x in data["list"]])

    # 3. Data Cleaning
    df = df.drop_duplicates().fillna(0)
    df["Date"] = pd.to_datetime(df["Date"])

    print(df.head())

    # 4. Data Analysis
    print("\nAverage Temperature:", df["Temp"].mean())
    print("Maximum Temperature:", df["Temp"].max())
    print("Minimum Temperature:", df["Temp"].min())

    # 5. Daily Aggregation
    print("\nDaily Average Temperature:")
    print(df.groupby(df["Date"].dt.date)["Temp"].mean())

    # 6. Temperature Line Plot
    df.plot(x="Date", y="Temp", title="Temperature Trend")
    plt.show()

    # 7. Humidity Bar Plot
    plt.bar(range(len(df)), df["Humidity"])
    plt.title("Humidity Levels")
    plt.show()

    # 8. Temperature vs Humidity
    plt.scatter(df["Temp"], df["Humidity"])
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.title("Temperature vs Humidity")
    plt.show()

    # 9. Correlation
    print("\nCorrelation:")
    print(df[["Temp", "Humidity", "Wind", "Rain"]].corr())

else:
    print("API Error:", data)
