# Indoor RF Analyzer — Two-Floor Wi-Fi Coverage Simulation

An interactive Streamlit dashboard that explores simulated Wi-Fi signal strength across a two-floor building.

### Features
- Adjustable router position and floor
- Predicted RSSI heatmaps for both floors
- Room-by-room coverage summaries
- Random Forest regression
- Downloadable CSV results

### Tools
Python, Streamlit, NumPy, Pandas, Matplotlib, Scikit-learn.

### Model results
On held-out synthetic test data:
- MAE: 3.113 dB
- RMSE: 3.800 dB

**Note:** The current version uses synthetic data and simplified room/floor loss assumptions. These metrics do not establish real-world Wi-Fi prediction accuracy.

### Run locally
Install dependencies with `pip install -r requirements.txt`, then run:

`python -m streamlit run app.py`
