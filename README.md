# 🚦 AI Accident Hotspot Prediction System

An AI-powered Road Safety and Accident Risk Prediction System built using Streamlit, Machine Learning, Weather APIs, Traffic Analysis, Route Intelligence, and AI Chatbot integration.

---

# 📌 Features

## 🚗 Accident Risk Prediction
- Predicts accident probability using Machine Learning.
- Displays Low / Medium / High risk zones.

## 🌦 Real-Time Weather Analysis
- Fetches live weather using OpenWeather API.
- Uses weather conditions in risk analysis.

## 🛣 Smart Route Analysis
- Calculates:
  - Distance
  - Travel time
  - Route information

## 🚶 Multi-Travel Mode Support
- Car
- Bike
- Walking
- Bus

## 🔊 Voice Alert System
- Gives voice warnings for dangerous routes.

## 🤖 AI Road Safety Chatbot
- AI assistant powered by Groq LLM.
- Answers:
  - Road safety queries
  - Traffic guidance
  - Weather-based suggestions
  - Safe travel timing

## 📊 Live Analytics Dashboard
- Temperature
- Humidity
- Weather conditions
- Accident probability

## 🎨 Modern UI
- Responsive modern dashboard UI using Streamlit custom CSS.

---

# 🛠 Technologies Used

- Python
- Streamlit
- Machine Learning
- Pandas
- Scikit-learn
- OpenWeather API
- OpenRouteService API
- Groq API
- gTTS
- Plotly

---

# 📂 Project Structure

```bash
AI_ROAD_DETECTION/
│
├── app.py
├── model.pkl
├── prepare_data.py
├── train_model.py
├── accident.csv
├── accidents_clean.csv
├── requirements.txt
├── .env
│
├── pages/
│   └── AI_Chatbot.py
│
├── visualization/
│
└── alert.mp3