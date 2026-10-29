
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# INDOOR RF ANALYZER - TWO-FLOOR SYNTHETIC SIMULATION
# All generated values are simulated, not real measurements.
# =========================================================

ROOM_LENGTH = 18.0
ROOM_WIDTH = 12.0
SEED = 42
SAMPLES_PER_ROOM = 100

# Configurable illustrative signal-loss assumptions
WALL_LOSS_DB = 8.0
FLOOR_LOSS_DB = 18.0
NOISE_STD_DB = 3.5

# Room format: (name, x_min, y_min, x_max, y_max)
ROOMS = {
    0: [
        ("Entrance Veranda", 0, 0, 4, 2),
        ("Bedroom 1", 0, 2, 4, 7),
        ("Bedroom 2", 0, 7, 4, 12),
        ("Hall", 4, 0, 12, 8),
        ("Passage", 4, 8, 12, 12),
        ("Kitchen", 12, 0, 18, 4),
        ("Garage", 12, 4, 18, 12),
    ],
    1: [
        ("Bedroom 3", 0, 0, 6, 5),
        ("Bedroom 4", 6, 0, 12, 5),
        ("Upstairs Landing", 0, 5, 12, 12),
        ("Balcony", 12, 0, 18, 3),
        ("Upper Passage", 12, 3, 18, 12),
    ],
}

# Assign a numeric code to each room for model input.
ROOM_NAMES = [
    room[0]
    for floor_rooms in ROOMS.values()
    for room in floor_rooms
]

ROOM_CODES = {
    name: index
    for index, name in enumerate(ROOM_NAMES)
}


def find_room(x, y, floor):
    """Return the room containing a coordinate."""
    for name, x1, y1, x2, y2 in ROOMS[int(floor)]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name

    return "Unassigned"


def calculate_room_loss(
    router_room, target_room, router_floor, target_floor
):
    """Apply simplified room and floor attenuation."""
    loss = 0.0

    if router_room != target_room:
        loss += WALL_LOSS_DB

    if router_floor != target_floor:
        loss += FLOOR_LOSS_DB

    return loss


def generate_data(
    router_x=8.0,
    router_y=4.0,
    router_floor=0
):
    """Generate reproducible synthetic RSSI samples."""
    rng = np.random.default_rng(SEED)

    router_room = find_room(
        router_x, router_y, router_floor
    )

    if router_room == "Unassigned":
        raise ValueError(
            "Router coordinates must be inside a defined room."
        )

    rows = []

    for floor, floor_rooms in ROOMS.items():
        for room_info in floor_rooms:
            name, x1, y1, x2, y2 = room_info

            x_values = rng.uniform(
                x1 + 0.05, x2 - 0.05, SAMPLES_PER_ROOM
            )
            y_values = rng.uniform(
                y1 + 0.05, y2 - 0.05, SAMPLES_PER_ROOM
            )

            room_loss = calculate_room_loss(
                router_room, name, router_floor, floor
            )

            vertical_distance = (
                3.0 if floor != router_floor else 0.0
            )

            distances = np.sqrt(
                (x_values - router_x) ** 2
                + (y_values - router_y) ** 2
                + vertical_distance ** 2
            )
            distances = np.maximum(distances, 0.5)

            # Illustrative log-distance model plus assumed losses.
            rssi_values = (
                -30
                - 22 * np.log10(distances)
                - room_loss
                + rng.normal(0, NOISE_STD_DB, SAMPLES_PER_ROOM)
            )

            for i in range(SAMPLES_PER_ROOM):
                rows.append({
                    "floor": floor,
                    "room": name,
                    "room_code": ROOM_CODES[name],
                    "x_m": x_values[i],
                    "y_m": y_values[i],
                    "distance_m": distances[i],
                    "room_loss_db": room_loss,
                    "rssi_dbm": rssi_values[i],
                })

    return pd.DataFrame(rows)


def run_analysis(
    router_x=8.0,
    router_y=4.0,
    threshold=-67,
    router_floor=0
):
    """Train the model and predict coverage on both floors."""

    router_room = find_room(
        router_x, router_y, router_floor
    )

    if router_room == "Unassigned":
        raise ValueError(
            "Choose router coordinates inside a defined room."
        )

    data = generate_data(
        router_x, router_y, router_floor
    )

    features = [
        "x_m",
        "y_m",
        "floor",
        "room_code",
        "distance_m",
        "room_loss_db",
    ]

    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=SEED
    )

    model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(train[features], train["rssi_dbm"])

    predictions = model.predict(test[features])

    mae = mean_absolute_error(
        test["rssi_dbm"], predictions
    )
    rmse = np.sqrt(mean_squared_error(
        test["rssi_dbm"], predictions
    ))

    metrics = pd.DataFrame([{
        "Model": "Random Forest",
        "MAE (dB)": round(mae, 3),
        "RMSE (dB)": round(rmse, 3),
    }])

    evaluation = test.copy()
    evaluation["Predicted RSSI (dBm)"] = predictions
    evaluation["Absolute error (dB)"] = np.abs(
        evaluation["rssi_dbm"] - predictions
    )

    # Generate a coverage grid for each floor.
    grid_x = np.linspace(0, ROOM_LENGTH, 100)
    grid_y = np.linspace(0, ROOM_WIDTH, 70)
    xx, yy = np.meshgrid(grid_x, grid_y)

    grid_rows = []

    for floor in ROOMS:
        for x, y in zip(xx.ravel(), yy.ravel()):
            room = find_room(x, y, floor)

            if room == "Unassigned":
                continue

            room_loss = calculate_room_loss(
                router_room, room, router_floor, floor
            )

            vertical_distance = (
                3.0 if floor != router_floor else 0.0
            )

            distance = np.sqrt(
                (x - router_x) ** 2
                + (y - router_y) ** 2
                + vertical_distance ** 2
            )
            distance = max(distance, 0.5)

            grid_rows.append({
                "floor": floor,
                "room": room,
                "room_code": ROOM_CODES[room],
                "x_m": x,
                "y_m": y,
                "distance_m": distance,
                "room_loss_db": room_loss,
            })

    grid = pd.DataFrame(grid_rows)

    grid["predicted_rssi_dbm"] = model.predict(
        grid[features]
    )

    grid["weak_coverage"] = (
        grid["predicted_rssi_dbm"] < threshold
    )

    return data, evaluation, metrics, grid


if __name__ == "__main__":
    data, evaluation, metrics, grid = run_analysis()

    data.to_csv("synthetic_rssi.csv", index=False)
    evaluation.to_csv("test_predictions.csv", index=False)
    metrics.to_csv("model_metrics.csv", index=False)
    grid.to_csv("coverage_grid.csv", index=False)

    print("\nINDOOR RF ANALYZER - TWO FLOOR SIMULATION")
    print("DATA TYPE: SYNTHETIC — NOT REAL MEASUREMENTS")
    print(f"Floors modelled: {len(ROOMS)}")
    print(f"Room and area zones: {len(ROOM_NAMES)}")
    print(f"Generated samples: {len(data)}")
    print(f"Held-out test samples: {len(evaluation)}")

    print("\nMODEL EVALUATION")
    print(metrics.to_string(index=False))

    print("\nCSV files saved:")
    print("- synthetic_rssi.csv")
    print("- test_predictions.csv")
    print("- model_metrics.csv")
    print("- coverage_grid.csv")
