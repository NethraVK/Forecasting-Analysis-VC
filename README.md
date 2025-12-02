# Event Forecasting Dashboard  
**Predicting event demand and staffing needs with data-driven insights**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)]()

---

## 📘 Table of Contents  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Why This Matters](#why-this-matters)  
4. [Tech Stack](#tech-stack)  
5. [How to Run](#how-to-run)   
7. [Using the App](#using-the-app)
8. [Common Issues & Fixes](#common-issues--fixes)
   
---

## Overview  
The **Event Forecasting Dashboard** uses machine learning models to forecast:

- Expected event demand  
- Expected staff requirements  
- Language-specific host/hostess shortages  
- Month-by-month demand spikes  

It uses a **Poisson regression model** enhanced with:

- Lag features  
- Rolling windows  
- Seasonality detection  
- Host availability integration  

To ensure realistic forecasting, the system includes:

- **Rolling-origin backtesting** (simulated month-by-month forecasting)  
- **Validation splits** for unseen months  

The Streamlit dashboard provides an interactive UI to explore forecasts, adjust staffing scenarios, and upload custom data.

---

## Features  

### Forecasting Engine  
- Poisson regression  
- Lag features  
- Rolling window features  
- Seasonality signals  
- Staff availability as input  
- Spike detection  

### Robust Model Validation  
- Rolling-origin backtesting  
- Temporal validation splits  
- Prevents overfitting and ensures real forecasting ability  

### Interactive Streamlit Dashboard  
- Demand vs Staff visualizer  
- Shortage indicators (highlighted in red)  
- "What-if" scenarios:
  - +X% events  
  - +X staff  
- Language-specific filters  
- Upload your own JSON dataset  
- Historical + forecast time-series charts  

---

## Why This Matters 

Event agencies often rely on guesswork or reactive planning. This tool helps:
- Identify shortages early
- Optimize staffing & reduce costs
- Forecast busy months
- Plan across languages
- Make data-driven decisions

---

## Tech Stack  
- **Python** (Pandas, Scikit-learn, Statsmodels, NumPy)  
- **Streamlit**  
- **Plotly / Matplotlib**  
- **MongoDB / JSON** data ingestion  

---

## How to Run  

### 1. Clone the repository  
```bash
git clone https://github.com/NethraVK/Forecasting-Analysis-VC.git
cd Forecasting-Analysis-VC
```
### 2. Create a virtual environment
#### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Optional MongoDB support
```bash
pip install pymongo
```
### 4. Run the Dashboard
```bash
streamlit run app.py
```
Access via: http://localhost:8501

---
## Using the App
- Load default JSON dataset or provide your own
- View forecasted demand vs staffing availability
- Adjust "what-if" sliders
- Filter by language
- Explore interactive historical + forecast charts
---
## Common Issues & Fixes
- ModuleNotFoundError → pip install -r requirements.txt
- Streamlit not launching → Ensure virtual environment is activated
- JSON file not found → Use absolute paths
