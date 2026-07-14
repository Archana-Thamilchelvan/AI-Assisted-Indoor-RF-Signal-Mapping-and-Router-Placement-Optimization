import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("wifi_distance.csv")

# Take absolute log distance
df["log_d"] = np.log10(df["Distance"])

# Prepare data
X = df[["log_d"]]
y = df["RSSI"]

# Train regression model
model = LinearRegression()
model.fit(X, y)

A = model.intercept_
n = -model.coef_[0] / 10

print("Estimated A (RSSI at 1m):", A)
print("Estimated Path Loss Exponent n:", n)

# Plot
plt.scatter(df["Distance"], df["RSSI"])
d_range = np.linspace(min(df["Distance"]), max(df["Distance"]), 100)
pred = model.predict(np.log10(d_range).reshape(-1,1))
plt.plot(d_range, pred)

plt.xlabel("Distance (m)")
plt.ylabel("RSSI (dBm)")
plt.title("Path Loss Model")
plt.show()