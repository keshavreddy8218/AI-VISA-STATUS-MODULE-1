import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
file_path = "visa_processing_dataset_10k.csv"

df = pd.read_csv(file_path)

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values count:")
print(df.isnull().sum())


df["application_date"].fillna(df["application_date"].mode()[0], inplace=True)
df["decision_date"].fillna(df["decision_date"].mode()[0], inplace=True)

df["country"].fillna("Unknown", inplace=True)
df["visa_type"].fillna("Unknown", inplace=True)
df["processing_office"].fillna("Unknown", inplace=True)



df["application_date"] = pd.to_datetime(df["application_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])


df["processing_days"] = (df["decision_date"] - df["application_date"]).dt.days

print("\nAfter calculating processing days:")
print(df.head())


df_encoded = pd.get_dummies(
    df,
    columns=["country", "visa_type", "processing_office"],
    drop_first=True
)

print("\nEncoded DataFrame:")
print(df_encoded.head())


X = df_encoded.drop(
    columns=["processing_days", "application_date", "decision_date"]
)
y = df_encoded["processing_days"]


model = LinearRegression()
model.fit(X, y)

print("\nModel training completed")


sample_input = X.iloc[0].values.reshape(1, -1)

predicted_days = model.predict(sample_input)
print("Predicted Processing Time:", predicted_days[0], "days")



print("\nStatistical Summary of Processing Days:")
print(df["processing_days"].describe())

sns.histplot(df["processing_days"], kde=True)
plt.title("Distribution of Visa Processing Days")
plt.xlabel("Processing Days")
plt.ylabel("Count")
plt.show()

sns.boxplot(x=df["processing_days"])
plt.title("Boxplot of Processing Days")
plt.show()
df["application_month"] = df["application_date"].dt.month
print(df[["application_date", "application_month"]].head())

df["season"] = df["application_month"].apply(
    lambda x: "Peak" if x in [1, 2, 12] else "Off-Peak"
)

print(df[["application_month", "season"]].head())

country_avg = df.groupby("country")["processing_days"].mean()
df["country_avg"] = df["country"].map(country_avg)

print("\nCountry-wise average processing days:")
print(country_avg.head())

visa_avg = df.groupby("visa_type")["processing_days"].mean()
df["visa_avg"] = df["visa_type"].map(visa_avg)

print("\nVisa-type average processing days:")
print(visa_avg)


corr_matrix = df[
    ["processing_days", "application_month", "country_avg", "visa_avg"]
].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)


sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.scatterplot(
    x="application_month",
    y="processing_days",
    data=df
)
plt.title("Processing Days vs Application Month")
plt.show()
