# Event Forecasting Dashboard  

### Predicting event demand and staffing needs with data-driven insights  

---

## 🔮 What I Built  

### 1. The Forecasting “Brain”  
I trained **Poisson regression models** to detect patterns in past event data — including **seasonality**, **demand spikes**, and **host availability**.  
To make the forecasts more realistic, I used **lag and rolling features** that help the model “see” how trends evolve over time.  

---

### 2. Reliable, Not Just Lucky  
To ensure the model could truly **predict the future (not just memorize the past)**, I implemented:  
- **Rolling-origin backtests** – testing the model month by month on unseen data  
- **Validation splits** – verifying forecasts on months the model hadn’t seen  

✅ **Result:** Forecasts that generalize beyond historical patterns.  

---

### 3. The Dashboard Everyone Sees  
Built with **Streamlit**, the dashboard turns predictions into actionable insights:  

- View **upcoming event demand** vs **available hostesses** — shortages flagged in red  
- Adjust **“what if” scenarios** (e.g., “What if we take 10% more events?” or “What if we add 5 new staff?”)  
- Filter by **language** to plan staffing more precisely  
- Plug in your **own JSON file path** to generate predictions dynamically  
- Visualize **historical vs forecast trends** with interactive charts  

---

## 💡 Why This Matters  
Event companies can now:  
- **Plan staffing ahead of time** instead of reacting last-minute  
- **Spot shortages early** and act proactively  
- **Optimize costs** by aligning resources with real demand  

👉 Turning **gut feeling** into **data-driven planning**.  

---

## 🧰 Tech Stack  
- **Python** (Pandas, NumPy, Scikit-learn, Statsmodels)  
- **Streamlit** for the dashboard  
- **Plotly / Matplotlib** for visualization  
- **MongoDB / JSON** for data handling  

---

## 🚀 How to Run  

1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/event-forecast-dashboard.git
   cd event-forecast-dashboard
