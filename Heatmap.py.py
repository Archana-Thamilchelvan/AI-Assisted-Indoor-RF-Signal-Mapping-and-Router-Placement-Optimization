import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# RSSI values from locations
data = np.array([
[-4, -10, -20],
[-37, -45, -55],
[-70, -80, -85]
])

plt.figure(figsize=(6,5))

sns.heatmap(data, annot=True, cmap="coolwarm")

plt.title("WiFi Signal Strength Heatmap (RSSI dBm)")
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.show()