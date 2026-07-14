import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 1. PATH LOSS PARAMETERS
# ==========================

A = -29.19   # RSSI at 1 meter (from your result)
n = 3.55     # Path loss exponent (from your result)

# ==========================
# 2. ROOM GRID DEFINITION
# ==========================

room_length = 18   # meters (your max distance)
room_width = 10    # assume width 10m

x = np.linspace(0.1, room_length, 50)
y = np.linspace(0.1, room_width, 50)

X, Y = np.meshgrid(x, y)

# ==========================
# 3. PATH LOSS FUNCTION
# ==========================

def rssi_model(d):
    return A - 10 * n * np.log10(d)

# ==========================
# 4. ROUTER OPTIMIZATION
# ==========================

best_score = -99999
best_position = (0, 0)

for rx in np.linspace(0.5, room_length, 20):
    for ry in np.linspace(0.5, room_width, 20):

        distance = np.sqrt((X - rx)**2 + (Y - ry)**2)
        distance[distance < 0.1] = 0.1

        predicted_rssi = rssi_model(distance)

        coverage_score = np.mean(predicted_rssi)

        if coverage_score > best_score:
            best_score = coverage_score
            best_position = (rx, ry)

print("Optimal Router Location (meters):")
print("X =", best_position[0])
print("Y =", best_position[1])
print("Average Predicted RSSI =", best_score)

# ==========================
# 5. FINAL HEATMAP
# ==========================

rx_opt, ry_opt = best_position
distance = np.sqrt((X - rx_opt)**2 + (Y - ry_opt)**2)
distance[distance < 0.1] = 0.1
final_rssi = rssi_model(distance)

plt.figure()
plt.imshow(final_rssi, extent=[0, room_length, 0, room_width], origin='lower')
plt.scatter(rx_opt, ry_opt)
plt.title("Optimized Router Placement Heatmap")
plt.xlabel("Room Length (m)")
plt.ylabel("Room Width (m)")
plt.colorbar(label="Predicted RSSI (dBm)")
plt.show()