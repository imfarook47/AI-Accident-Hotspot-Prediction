import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
import requests
import os
import openrouteservice
import time
import uuid


from dotenv import load_dotenv
from streamlit_folium import st_folium
from gtts import gTTS
from playsound import playsound

# Import heatmap function
from visualization.heatmap import generate_heatmap

# ----------------------------------
# Load Environment Variables
# ----------------------------------
load_dotenv()


API_KEY = os.getenv("OPENWEATHER_API_KEY")
ORS_API_KEY = os.getenv("ORS_API_KEY")

# ----------------------------------
# OpenRouteService Client
# ----------------------------------
client = openrouteservice.Client(
    key=ORS_API_KEY
)

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AI Accident Hotspot Prediction",
    layout="wide"
)

# ----------------------------------
# MODERN UI CSS
# ----------------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background-color: #f1f5f9;
}

.stApp {
    background-color: #f1f5f9;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* ================================= */
/* INPUT BOXES */
/* ================================= */

.stTextInput input,
.stDateInput input,
.stTimeInput input {

    background-color: white !important;

    color: black !important;

    border-radius: 12px !important;

    border: 2px solid #cbd5e1 !important;

    padding: 12px !important;

    font-size: 16px !important;
}

/* ================================= */
/* TIME INPUT FULL FIX */
/* ================================= */

input[type="time"] {

    color: black !important;

    background-color: white !important;

    -webkit-text-fill-color: black !important;

    opacity: 1 !important;

    caret-color: black !important;
}

.stTimeInput div[data-baseweb="input"] {

    background-color: white !important;

    border-radius: 12px !important;

    border: 2px solid #cbd5e1 !important;
}

.stTimeInput input {

    color: black !important;

    -webkit-text-fill-color: black !important;

    opacity: 1 !important;

    background-color: white !important;

    font-size: 18px !important;

    font-weight: bold !important;
}

.stTimeInput div {

    color: black !important;
}

/* Time icon */

input[type="time"]::-webkit-calendar-picker-indicator {

    filter: invert(0);
}

/* Time input height */

input[type="time"] {

    min-height: 45px;
}


/* DATE FIX */

.stDateInput input {

    color: black !important;

    background-color: white !important;

    -webkit-text-fill-color: black !important;
}

/* Placeholder */

input::placeholder {

    color: gray !important;
}

/* Icons */

svg {

    color: black !important;
}

/* TITLES */

h1, h2, h3 {

    color: #1e293b;
}

/* METRIC CARDS */

[data-testid="metric-container"] {

    background: white;

    padding: 20px;

    border-radius: 15px;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);

    border: 1px solid #e2e8f0;
}

/* RISK CARDS */

.risk-high {

    background: #fee2e2;

    border-left: 8px solid red;

    padding: 20px;

    border-radius: 12px;

    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.risk-medium {

    background: #fef3c7;

    border-left: 8px solid orange;

    padding: 20px;

    border-radius: 12px;

    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.risk-low {

    background: #dcfce7;

    border-left: 8px solid green;

    padding: 20px;

    border-radius: 12px;

    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* BUTTONS */

.stButton button {

    background: linear-gradient(
        90deg,
        #2563eb,
        #1d4ed8
    );

    color: white !important;

    border-radius: 12px;

    border: none;

    padding: 12px 20px;

    font-size: 16px;

    font-weight: bold;

    transition: 0.3s;
}

.stButton button:hover {

    transform: scale(1.03);

    background: linear-gradient(
        90deg,
        #1d4ed8,
        #1e40af
    );
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# TITLE
# ----------------------------------
st.title("🚦 AI-Based Accident Hotspot Prediction System")

st.markdown("""
Predict accident risk using:

- Machine Learning
- Real-time Weather API
- Dynamic Route Analysis
- AI Risk Detection
- Voice Alert System
""")

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:

        model = pickle.load(file)

    return model

model = load_model()

# ----------------------------------
# VOICE ALERT FUNCTION
# ----------------------------------
def speak_alert(message):

    try:

        filename = f"alert_{uuid.uuid4()}.mp3"

        tts = gTTS(
            text=message,
            lang='en'
        )

        tts.save(filename)

        playsound(filename)

        time.sleep(1)

        os.remove(filename)

    except Exception as e:

        st.warning(
            f"Voice alert error: {e}"
        )

# ----------------------------------
# WEATHER FUNCTION
# ----------------------------------
def get_weather(city):

    try:

        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={API_KEY}&units=metric"
        )

        response = requests.get(url)

        data = response.json()

        temperature = data["main"]["temp"]

        weather_condition = data["weather"][0]["main"]

        humidity = data["main"]["humidity"]

        wind_speed = data["wind"]["speed"]

        return (
            temperature,
            weather_condition,
            humidity,
            wind_speed
        )

    except:

        return 30, "Clear", 50, 2

# ----------------------------------
# ROUTE FUNCTION
# ----------------------------------
def get_route(start, end):

    try:

        start_geo = client.pelias_search(start)

        end_geo = client.pelias_search(end)

        start_coords = (
            start_geo['features'][0]
            ['geometry']['coordinates']
        )

        end_coords = (
            end_geo['features'][0]
            ['geometry']['coordinates']
        )

        route = client.directions(
            coordinates=[
                start_coords,
                end_coords
            ],
            profile='driving-car',
            format='geojson'
        )

        return route

    except Exception as e:

        st.error(f"Route Error: {e}")

        return None

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.title("🗺 Travel Details")

from_location = st.sidebar.text_input(
    "From Location",
    value="Madurai"
)

to_location = st.sidebar.text_input(
    "To Location",
    value="Kodaikanal"
)

selected_date = st.sidebar.date_input(
    "Select Travel Date"
)

time_str = st.sidebar.text_input(
    "Select Travel Time (HH:MM)",
    value="08:00"
)

try:

    selected_time = pd.to_datetime(
        time_str,
        format="%H:%M"
    ).time()

except:

    st.sidebar.error(
        "Enter time in HH:MM format"
    )

    st.stop()


# ----------------------------------
# WEATHER FETCH
# ----------------------------------
temperature, weather_condition, humidity, wind_speed = get_weather(
    from_location
)

# Sidebar Weather
st.sidebar.success(
    f"🌡 Temperature: {temperature} °C"
)

st.sidebar.info(
    f"☁ Weather: {weather_condition}"
)

st.sidebar.info(
    f"💧 Humidity: {humidity}%"
)

st.sidebar.info(
    f"💨 Wind Speed: {wind_speed} m/s"
)

# ----------------------------------
# FEATURE EXTRACTION
# ----------------------------------
hour = selected_time.hour

is_weekend = 1 if selected_date.weekday() >= 5 else 0

# Visibility
if weather_condition in ["Fog", "Mist", "Haze"]:

    visibility = 3

elif weather_condition in ["Rain", "Thunderstorm"]:

    visibility = 5

else:

    visibility = 10

# ----------------------------------
# DASHBOARD METRICS
# ----------------------------------
# ----------------------------------
# TRAVEL MODE ANALYSIS
# ----------------------------------

st.subheader("🚘 Travel Modes")

def get_travel_info(start, end, profile):

    try:

        start_geo = client.pelias_search(start)

        end_geo = client.pelias_search(end)

        start_coords = (
            start_geo['features'][0]
            ['geometry']['coordinates']
        )

        end_coords = (
            end_geo['features'][0]
            ['geometry']['coordinates']
        )

        route = client.directions(
            coordinates=[
                start_coords,
                end_coords
            ],
            profile=profile,
            format='geojson'
        )

        summary = (
            route['features'][0]
            ['properties']['summary']
        )

        distance = round(
            summary['distance'] / 1000,
            2
        )

        duration = round(
            summary['duration'] / 3600,
            2
        )

        return distance, duration

    except:

        return 0, 0

car_distance, car_time = get_travel_info(
    from_location,
    to_location,
    "driving-car"
)

bike_distance, bike_time = get_travel_info(
    from_location,
    to_location,
    "cycling-regular"
)

walk_distance, walk_time = get_travel_info(
    from_location,
    to_location,
    "foot-walking"
)

bus_time = round(car_time * 1.2, 2)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "🚗 Car",
        f"{car_time} hr",
        f"{car_distance} KM"
    )

with c2:

    st.metric(
        "🏍 Bike",
        f"{bike_time} hr",
        f"{bike_distance} KM"
    )

with c3:

    st.metric(
        "🚶 Walk",
        f"{walk_time} hr",
        f"{walk_distance} KM"
    )

with c4:

    st.metric(
        "🚌 Bus",
        f"{bus_time} hr",
        f"{car_distance} KM"
    )

# ----------------------------------
# PREDICTION SECTION
# ----------------------------------
st.subheader("🎯 Accident Risk Prediction")

if "risk_percent" not in st.session_state:

    st.session_state.risk_percent = None

    st.session_state.route = None

# ----------------------------------
# PREDICT BUTTON
# ----------------------------------
if st.button("🔍 Predict Accident Risk"):

    with st.spinner(
        "🤖 AI analyzing accident probability..."
    ):

        features = np.array([ 
            [
                hour,
                temperature,
                visibility,
                is_weekend
            ]
        ])

        probability = model.predict_proba(features)[0][1]

        risk_percent = round(probability * 100, 2)

        st.session_state.risk_percent = risk_percent

        st.session_state.route = (
            f"{from_location} ➡ {to_location}"
        )

# ----------------------------------
# DISPLAY RESULTS
# ----------------------------------
if st.session_state.risk_percent is not None:

    risk_percent = st.session_state.risk_percent

    st.markdown(
        f"# 🛣 Route: {st.session_state.route}"
    )

    # HIGH RISK
    if risk_percent > 70:

        st.markdown(f"""
        <div class="risk-high">
        <h2>⚠ HIGH RISK ZONE</h2>
        <h3>Accident Probability: {risk_percent}%</h3>
        </div>
        """, unsafe_allow_html=True)

        speak_alert(
            "Warning! High accident risk detected. Drive carefully."
        )

    # MEDIUM RISK
    elif risk_percent > 40:

        st.markdown(f"""
        <div class="risk-medium">
        <h2>⚠ MEDIUM RISK ZONE</h2>
        <h3>Accident Probability: {risk_percent}%</h3>
        </div>
        """, unsafe_allow_html=True)

        speak_alert(
            "Moderate accident risk detected. Please stay alert."
        )

    # LOW RISK
    else:

        st.markdown(f"""
        <div class="risk-low">
        <h2>✅ LOW RISK ZONE</h2>
        <h3>Accident Probability: {risk_percent}%</h3>
        </div>
        """, unsafe_allow_html=True)

        speak_alert(
            "Travel conditions appear safe."
        )

    # Progress Bar
    st.progress(int(risk_percent))



# ----------------------------------
# HEATMAP SECTION
# ----------------------------------
st.subheader("🗺 Accident Hotspot Heatmap")

try:

    df = pd.read_csv("accidents_clean.csv")

    df = df[
        (df["Latitude"] > 8) &
        (df["Latitude"] < 11.5) &
        (df["Longitude"] > 76) &
        (df["Longitude"] < 79)
    ]

    m = generate_heatmap(
        df,
        selected_hour=hour,
        radius=15,
        blur=10
    )

    st_folium(
        m,
        width=1000,
        height=500
    )

except Exception as e:

    st.warning(
        "Heatmap could not be generated."
    )

    st.write(e)

# ----------------------------------
# ROUTE VISUALIZATION
# ----------------------------------
st.subheader("🛣 Dynamic Route Visualization")

if from_location and to_location:

    route_data = get_route(
        from_location,
        to_location
    )

    if route_data:

        coordinates = (
            route_data['features'][0]
            ['geometry']['coordinates']
        )

        route_points = [
            [coord[1], coord[0]]
            for coord in coordinates
        ]

        route_map = folium.Map(
            location=route_points[0],
            zoom_start=8
        )

        # Route Line
        folium.PolyLine(
            route_points,
            color="blue",
            weight=6,
            opacity=0.8
        ).add_to(route_map)

        # Start Marker
        folium.Marker(
            route_points[0],
            tooltip="Start",
            popup=from_location,
            icon=folium.Icon(color="green")
        ).add_to(route_map)

        # Destination Marker
        folium.Marker(
            route_points[-1],
            tooltip="Destination",
            popup=to_location,
            icon=folium.Icon(color="red")
        ).add_to(route_map)

        # Danger Marker
        middle_point = route_points[
            len(route_points)//2
        ]

        folium.Marker(
            middle_point,
            tooltip="Danger Zone",
            popup="⚠ Accident-prone area",
            icon=folium.Icon(
                color="orange",
                icon="warning-sign"
            )
        ).add_to(route_map)

        st_folium(
            route_map,
            width=1000,
            height=500
        )

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")

st.markdown("""
### 👨‍💻 Developed By

- Mohamed Farook T  
- Nithish Kumar K  
- Negeskar V
""")