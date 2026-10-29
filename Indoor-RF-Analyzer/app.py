
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rf_engine import (
    run_analysis,
    ROOMS,
    ROOM_LENGTH,
    ROOM_WIDTH,
    WALL_LOSS_DB,
    FLOOR_LOSS_DB,
)

st.set_page_config(
    page_title="Indoor RF Analyzer",
    page_icon="📶",
    layout="wide",
)

st.title("📶 Indoor RF Analyzer")
st.caption(
    "Two-floor Wi-Fi coverage simulation | "
    "Synthetic data — not real-world measurements"
)

st.info(
    "This is an illustrative simulation. Wall and floor losses "
    "are simplified assumptions, not calibrated measurements."
)

# ---------------- SIDEBAR CONTROLS ----------------

st.sidebar.header("Simulation Settings")

router_floor_label = st.sidebar.selectbox(
    "Router floor",
    ["Ground Floor", "First Floor"],
)
router_floor = (
    0 if router_floor_label == "Ground Floor" else 1
)

router_x = st.sidebar.slider(
    "Router X position (m)",
    min_value=0.5,
    max_value=17.5,
    value=8.0,
    step=0.5,
)

router_y = st.sidebar.slider(
    "Router Y position (m)",
    min_value=0.5,
    max_value=11.5,
    value=4.0,
    step=0.5,
)

threshold = st.sidebar.slider(
    "Weak-signal threshold (dBm)",
    min_value=-85,
    max_value=-45,
    value=-67,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Illustrative loss assumptions")

st.sidebar.write(f"Room-change penalty: {WALL_LOSS_DB:.1f} dB")
st.sidebar.write(f"Other-floor penalty: {FLOOR_LOSS_DB:.1f} dB")

# ---------------- RUN SIMULATION ----------------

data, evaluation, metrics, grid = run_analysis(
    router_x=router_x,
    router_y=router_y,
    threshold=threshold,
    router_floor=router_floor,
)

router_room = next(
    (
        name
        for name, x1, y1, x2, y2 in ROOMS[router_floor]
        if x1 <= router_x <= x2
        and y1 <= router_y <= y2
    ),
    "Unassigned",
)

st.caption(
    f"Router location: {router_floor_label} · "
    f"{router_room} · X={router_x:.1f} m, Y={router_y:.1f} m"
)

# ---------------- SUMMARY METRICS ----------------

floor_grid = grid[grid["floor"] == router_floor]
weak_percent = (
    100 * floor_grid["weak_coverage"].mean()
    if not floor_grid.empty else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Floors", "2")
c2.metric("Room/area zones", "12")
c3.metric("Synthetic samples", f"{len(data):,}")
c4.metric(
    "Weak coverage in selected floor",
    f"{weak_percent:.1f}%",
)

# ---------------- FLOOR SELECTOR ----------------

selected_floor_label = st.radio(
    "View coverage map",
    ["Ground Floor", "First Floor"],
    horizontal=True,
)

selected_floor = (
    0 if selected_floor_label == "Ground Floor" else 1
)

floor_grid = grid[grid["floor"] == selected_floor].copy()

# ---------------- ROOM LAYOUT + HEATMAP ----------------

st.subheader(f"{selected_floor_label} — Predicted Wi-Fi coverage")

fig, ax = plt.subplots(figsize=(12, 6))

# Create a rectangular RSSI grid from the predicted points.
pivot = floor_grid.pivot_table(
    index="y_m",
    columns="x_m",
    values="predicted_rssi_dbm",
    aggfunc="mean",
)

im = ax.imshow(
    pivot.values,
    origin="lower",
    extent=[
        pivot.columns.min(),
        pivot.columns.max(),
        pivot.index.min(),
        pivot.index.max(),
    ],
    aspect="auto",
    cmap="RdYlGn",
    vmin=-90,
    vmax=-35,
    interpolation="nearest",
)

colorbar = fig.colorbar(im, ax=ax)
colorbar.set_label("Predicted RSSI (dBm)")

# Draw the room outlines and names.
for name, x1, y1, x2, y2 in ROOMS[selected_floor]:
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )
    ax.text(
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        name,
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        bbox=dict(
            facecolor="white",
            alpha=0.72,
            edgecolor="none",
            pad=2,
        ),
    )

# Show router only on the floor where it is located.
if selected_floor == router_floor:
    ax.scatter(
        router_x,
        router_y,
        marker="*",
        s=250,
        color="blue",
        edgecolors="white",
        linewidths=1,
        label="Router",
        zorder=5,
    )
    ax.legend(loc="upper right")

ax.set_xlim(0, ROOM_LENGTH)
ax.set_ylim(0, ROOM_WIDTH)
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_title("Synthetic signal-strength map with room boundaries")

st.pyplot(fig)
plt.close(fig)

# ---------------- ROOM COVERAGE SUMMARY ----------------

st.subheader("Room-by-room coverage summary")

room_summary = (
    floor_grid.groupby("room")
    .agg(
        Average_RSSI_dBm=("predicted_rssi_dbm", "mean"),
        Weak_Grid_Points=("weak_coverage", "sum"),
        Total_Grid_Points=("weak_coverage", "size"),
    )
    .reset_index()
)

room_summary["Weak_Points_Percent"] = (
    100
    * room_summary["Weak_Grid_Points"]
    / room_summary["Total_Grid_Points"]
).round(1)

room_summary["Average_RSSI_dBm"] = (
    room_summary["Average_RSSI_dBm"].round(2)
)

room_summary = room_summary.rename(
    columns={
        "room": "Room",
        "Average_RSSI_dBm": "Average predicted RSSI (dBm)",
        "Weak_Grid_Points": "Weak grid points",
        "Total_Grid_Points": "Total grid points",
        "Weak_Points_Percent": "Weak points (%)",
    }
)

st.dataframe(room_summary, hide_index=True, width="stretch")

# ---------------- MODEL EVALUATION ----------------

with st.expander("Model evaluation and test predictions"):
    st.write(
        "Metrics are calculated on held-out synthetic samples. "
        "They do not establish real-world Wi-Fi accuracy."
    )
    st.dataframe(metrics, hide_index=True, width="stretch")

    st.subheader("Test predictions")
    st.dataframe(
        evaluation.head(30),
        hide_index=True,
        width="stretch",
    )

with st.expander("Explore generated data"):
    st.dataframe(data.head(30), hide_index=True, width="stretch")

# ---------------- DOWNLOADS ----------------

st.subheader("Download simulation data")

d1, d2, d3, d4 = st.columns(4)

d1.download_button(
    "Download synthetic RSSI",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_rssi.csv",
    mime="text/csv",
)

d2.download_button(
    "Download test predictions",
    data=evaluation.to_csv(index=False).encode("utf-8"),
    file_name="test_predictions.csv",
    mime="text/csv",
)

d3.download_button(
    "Download model metrics",
    data=metrics.to_csv(index=False).encode("utf-8"),
    file_name="model_metrics.csv",
    mime="text/csv",
)

d4.download_button(
    "Download coverage grid",
    data=grid.to_csv(index=False).encode("utf-8"),
    file_name="coverage_grid.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Indoor RF Analyzer · Educational simulation · "
    "Synthetic data only"
)
