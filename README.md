# ⚡ AI-Powered Energy Consumption Forecasting System

## 🚀 Live Demo
🔗 Streamlit App: https://your-app-link.streamlit.app  
*(Add your link after deployment)*

---

## 📌 Overview

This project is an end-to-end **AI-powered energy consumption forecasting system** that predicts future energy usage using historical data.

It simulates real-world scenarios where industries like power grids, smart cities, and data centers need accurate energy forecasting to optimize operations and reduce costs.

The system includes:
- Data simulation
- Feature engineering
- Machine learning model
- Forecasting system
- Interactive Streamlit dashboard

---

## 🎯 Problem Statement

Accurate energy consumption forecasting helps organizations:

- Reduce operational costs  
- Optimize resource allocation  
- Prevent energy wastage  
- Improve demand planning  

---

## 🧠 Solution Approach

1. Generated synthetic time-series dataset
2. Applied feature engineering:
   - Hour, Day, Month
   - Lag features (previous values)
   - Rolling averages
3. Trained a **Random Forest Regression model**
4. Evaluated using:
   - RMSE
   - R² Score
5. Built an interactive dashboard using Streamlit

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Plotly (interactive charts)
- Streamlit (dashboard)
- Joblib (model saving)

---

## 📊 Features

✔ Energy consumption prediction  
✔ Feature engineering pipeline  
✔ Model evaluation metrics  
✔ Interactive dashboard  
✔ Dark-themed UI  
✔ Upload custom dataset  
✔ Live prediction button  
✔ 24-hour future forecasting  
✔ Download predictions  

---

## 📁 Project Structure
AI-Energy-Forecasting-System/
│
├── data/
├── models/
├── outputs/
├── images/
├── src/
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── visualization.py
│ ├── save_model.py
│
├── app.py
├── main.py
├── README.md
├── requirements.txt


---

## ⚙️ Installation

```bash
git clone https://github.com/sankeerth100/AI-Energy-Forecasting-System.git
cd AI-Energy-Forecasting-System
pip install -r requirements.txt

Run ML pipeline
python main.py

Run Streamlit App
python -m streamlit run app.py
