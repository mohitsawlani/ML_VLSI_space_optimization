# ML-Based Space Optimization in VLSI Chips

## Overview
This project uses a machine learning regression model to predict spatial density in VLSI chip layouts. The goal is to identify congestion regions and improve space utilization during the placement stage.

## Method
The chip layout is divided into grid regions and features such as cell count and cell area are extracted. A Linear Regression model is trained to predict density distribution across the chip.

## Tools Used
- Python
- NumPy
- Scikit-learn
- Matplotlib

## Project Files
- ml_density_simple.py – Machine learning model implementation
- X.npy, y.npy – Training dataset
- real_density_map.png – Density map visualization

## Author
Mohit Sawlani
B.Tech Microelectronics & VLSI  
NIT Kurukshetra
