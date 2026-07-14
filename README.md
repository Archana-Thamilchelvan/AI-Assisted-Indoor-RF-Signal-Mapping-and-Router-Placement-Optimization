# AI-Assisted Indoor RF Signal Mapping and Router Placement Optimization

## Project Overview
This project uses an ESP32-based wireless node to collect WiFi RSSI measurements at multiple indoor locations and analyze signal quality inside a building.

The collected data is processed using Python to generate signal heatmaps, perform statistical analysis, predict signal strength at unmeasured locations, and recommend optimal router placement.

## Features
- Real-time RSSI measurement using ESP32
- Indoor WiFi signal strength mapping
- Signal quality classification
- Statistical analysis of RSSI stability
- Heatmap visualization
- Regression-based signal prediction
- Router placement optimization

## Hardware Used
- ESP32 Development Board
- WiFi Router
- Laptop for data collection and processing

## Software Used
- Arduino IDE
- Python
- NumPy
- Pandas
- Matplotlib

## Working Methodology
1. ESP32 connects to the target WiFi network.
2. RSSI values are collected at multiple indoor locations.
3. Data is transmitted to the laptop and stored in CSV format.
4. Python performs statistical analysis on collected samples.
5. A regression model predicts RSSI values for unmeasured locations.
6. Heatmaps are generated to visualize coverage distribution.
7. The system recommends optimal router placement based on signal behavior.

## Repository Contents
- Dataset files
- Regression models
- Heatmap generation scripts
- Path loss analysis
- Optimization results

## Results
The system successfully generated indoor signal maps and provided router placement recommendations to improve wireless coverage and reduce weak signal regions.

## Future Improvements
- Multi-floor indoor mapping
- Real-time cloud dashboard
- Support for multiple access points
- Advanced machine learning models
