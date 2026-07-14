import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------
# LOAD YOUR DATA
# ---------------------------------

df = pd.read_csv("wifi_distance.csv")

distance = df["Distance"].values.reshape(-1,1)
rssi = df["RSSI"].values

# Log-distance feature
log_distance = np.log10(distance)

# ---------------------------------
# ML MODEL (Regression)
# ---------------------------------

model = LinearRegression()
model.fit(log_distance, rssi)

A = model.intercept_
n = -model.coef_[0] / 10

print("ML Estimated A:", A)
print("ML Estimated Path Loss Exponent n:", n)

# ---------------------------------
# ROOM GRID
# ---------------------------------

room_length = 18
room_width = 10

x = np.linspace(0, room_length, 100)
y = np.linspace(0, room_width, 100)
X, Y = np.meshgrid(x, y)

# Try all possible router positions
best_score = -999999
best_position = (0,0)

for rx in np.linspace(0, room_length, 30):
    for ry in np.linspace(0, room_width, 30):
        
        dist = np.sqrt((X - rx)**2 + (Y - ry)**2)
        dist[dist < 0.1] = 0.1
        
        pred_rssi = model.predict(np.log10(dist).reshape(-1,1))
        avg_rssi = np.mean(pred_rssi)
        
        if avg_rssi > best_score:
            best_score = avg_rssi
            best_position = (rx, ry)

print("\nOptimal Router Position (ML Based):", best_position)

# ---------------------------------
# FINAL HEATMAP
# ---------------------------------

rx, ry = best_position
dist = np.sqrt((X - rx)**2 + (Y - ry)**2)
dist[dist < 0.1] = 0.1

Z = model.predict(np.log10(dist).reshape(-1,1))
Z = Z.reshape(X.shape)

plt.figure()
plt.imshow(Z, extent=[0,room_length,0,room_width], origin='lower')
plt.colorbar(label="Predicted RSSI (dBm)")
plt.scatter(rx, ry)
plt.title("ML Based Router Placement Optimization")
plt.xlabel("Room Length (m)")
plt.ylabel("Room Width (m)")
plt.show()