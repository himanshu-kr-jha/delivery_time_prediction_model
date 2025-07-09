import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import random
from streamlit_folium import st_folium
import folium

# Page configuration
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for classic styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #ffa500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric {
        text-align: center;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        min-width: 120px;
    }
    
    .sidebar-header {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🚴 Food Delivery Time Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Predict delivery times using location, weather, and traffic data</p>', unsafe_allow_html=True)

# Initialize session state
if 'user_lat' not in st.session_state:
    st.session_state.user_lat = None
if 'user_lon' not in st.session_state:
    st.session_state.user_lon = None
if 'pickup_location' not in st.session_state:
    st.session_state.pickup_location = None

# API Keys
GOOGLE_MAPS_API_KEY = "AIzaSyDXMAJZyaokGu9VpSKIEqmtT0tIDCNF8M4"
OPENWEATHER_API_KEY = "cfe7e20fe0490c79c2edc40c2fcbe6a4"

# Predefined pickup locations
pickup_locations = [
    (12.9716, 77.5946, "MG Road", "🏪"),
    (12.9352, 77.6146, "Koramangala", "🍕"),
    (12.9279, 77.6271, "HSR Layout", "🍔"),
    (13.0358, 77.5970, "Malleshwaram", "🍜"),
    (12.9490, 77.7000, "Whitefield", "🥘")
]

def simplify_weather(weather_main):
    weather_main = weather_main.lower()
    if weather_main in ["clear"]:
        return "Clear"
    elif weather_main in ["clouds", "mist", "haze", "fog", "smoke", "dust", "sand", "ash"]:
        return "Foggy"
    elif weather_main in ["rain", "drizzle", "thunderstorm"]:
        return "Rainy"
    elif weather_main in ["snow"]:
        return "Snowy"
    elif weather_main in ["squall", "tornado"]:
        return "Windy"
    else:
        return "Clear"

def categorize_traffic_level(traffic_duration_text, normal_duration_text):
    """
    Categorize traffic level based on traffic duration vs normal duration
    """
    try:
        # Extract minutes from duration text (e.g., "15 mins" or "1 hour 5 mins")
        def extract_minutes(duration_text):
            if not duration_text or duration_text == "Unknown":
                return None
            
            duration_text = duration_text.lower()
            total_minutes = 0
            
            # Extract hours
            if "hour" in duration_text:
                hours = int(duration_text.split("hour")[0].strip().split()[-1])
                total_minutes += hours * 60
            
            # Extract minutes
            if "min" in duration_text:
                # Handle cases like "15 mins" or "1 hour 5 mins"
                mins_part = duration_text.split("min")[0]
                if "hour" in mins_part:
                    mins_part = mins_part.split("hour")[1]
                mins = int(mins_part.strip().split()[-1])
                total_minutes += mins
            
            return total_minutes
        
        traffic_minutes = extract_minutes(traffic_duration_text)
        normal_minutes = extract_minutes(normal_duration_text)
        
        if traffic_minutes is None or normal_minutes is None:
            return "Medium", None  # Default fallback
        
        # Calculate delay ratio
        delay_ratio = traffic_minutes / normal_minutes if normal_minutes > 0 else 1
        
        # Categorize based on delay
        if delay_ratio <= 1.2:  # Up to 20% delay
            return "Low", normal_minutes
        elif delay_ratio <= 1.5:  # 20-50% delay
            return "Medium", normal_minutes
        else:  # More than 50% delay
            return "High", normal_minutes
            
    except Exception:
        return "Medium", None  # Default fallback on any parsing error

def calculate_total_time(normal_duration_minutes, prep_time):
    """
    Calculate total estimated time = normal travel time + preparation time
    """
    if normal_duration_minutes is None:
        return None
    return normal_duration_minutes + int(prep_time)

# Sidebar for user inputs
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛠️ Configuration</div>', unsafe_allow_html=True)
    
    vehicle_type = st.selectbox("🚗 Vehicle Type", ["Bike", "Scooter", "Car"], key="vehicle")
    prep_time = st.slider("⏱️ Preparation Time (minutes)", 0, 60, 10,step=1,key="prep")
    experience = st.slider("👨‍💼 Courier Experience (years)", 0, 20, 2,step=1, key="exp")
    
    st.markdown("---")
    st.markdown("### 📍 How to use:")
    st.markdown("1. Click on the map to select delivery location")
    st.markdown("2. Adjust settings in sidebar")
    st.markdown("3. Get instant delivery prediction")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📍 Select Your Delivery Location")
    st.markdown("Click on the map to choose where you want your food delivered")
    
    # Default map center (Bangalore)
    default_location = [12.921916199187342, 77.67635277059242]
    
    # Create folium map
    m = folium.Map(location=default_location, zoom_start=13, tiles='OpenStreetMap')
    
    # Add pickup locations as markers
    for lat, lon, name, emoji in pickup_locations:
        folium.Marker(
            [lat, lon],
            popup=f"{emoji} {name} (Restaurant)",
            tooltip=f"Click to see {name}",
            icon=folium.Icon(color='red', icon='cutlery', prefix='fa')
        ).add_to(m)
    
    # Add user location marker if selected
    if st.session_state.user_lat and st.session_state.user_lon:
        folium.Marker(
            [st.session_state.user_lat, st.session_state.user_lon],
            popup="📍 Your Delivery Location",
            tooltip="Delivery Location",
            icon=folium.Icon(color='blue', icon='home', prefix='fa')
        ).add_to(m)
    
    # Add click functionality
    m.add_child(folium.LatLngPopup())
    
    # Display map
    map_data = st_folium(m, height=400, width=None)
    
    # Handle map clicks
    if map_data and map_data.get("last_clicked"):
        st.session_state.user_lat = map_data["last_clicked"]["lat"]
        st.session_state.user_lon = map_data["last_clicked"]["lng"]
        # Auto-refresh by rerunning
        st.rerun()

with col2:
    if st.session_state.user_lat and st.session_state.user_lon:
        st.markdown("### 📊 Delivery Information")
        
        # Select random pickup location
        pickup_lat, pickup_lon, pickup_name, pickup_emoji = random.choice(pickup_locations)
        st.session_state.pickup_location = (pickup_lat, pickup_lon, pickup_name, pickup_emoji)
        
        # Display location info
        st.markdown(f"""
        <div class="info-card">
            <strong>{pickup_emoji} Pickup:</strong> {pickup_name}<br>
            <strong>📍 Delivery:</strong> {st.session_state.user_lat:.4f}, {st.session_state.user_lon:.4f}
        </div>
        """, unsafe_allow_html=True)
        
        # Get distance and traffic data
        try:
            distance_url = (
                f"https://maps.googleapis.com/maps/api/distancematrix/json"
                f"?origins={pickup_lat},{pickup_lon}"
                f"&destinations={st.session_state.user_lat},{st.session_state.user_lon}"
                f"&departure_time=now"
                f"&key={GOOGLE_MAPS_API_KEY}"
            )
            distance_response = requests.get(distance_url).json()
            
            if distance_response["status"] == "OK":
                element = distance_response["rows"][0]["elements"][0]
                distance_km = element["distance"]["value"] / 1000
                traffic_duration = element.get("duration_in_traffic", {}).get("text", "Unknown")
                normal_duration = element.get("duration", {}).get("text", "Unknown")
                
                # Categorize traffic level and get normal duration in minutes
                traffic_level, normal_duration_minutes = categorize_traffic_level(traffic_duration, normal_duration)
                
                # Calculate total estimated time
                calculated_time = calculate_total_time(normal_duration_minutes, prep_time)
                
                # Get weather data
                weather_url = (
                    f"https://api.openweathermap.org/data/2.5/weather"
                    f"?lat={st.session_state.user_lat}&lon={st.session_state.user_lon}&appid={OPENWEATHER_API_KEY}"
                )
                weather_response = requests.get(weather_url).json()
                weather_main = simplify_weather(weather_response["weather"][0]["main"])
                
                # Determine time of day
                hour = datetime.now().hour
                if 5 <= hour < 12:
                    time_of_day = "Morning"
                elif 12 <= hour < 17:
                    time_of_day = "Afternoon"
                elif 17 <= hour < 21:
                    time_of_day = "Evening"
                else:
                    time_of_day = "Night"
                
                # Display metrics
                st.markdown("### 📈 Current Conditions")
                
                # Get traffic level color and emoji
                traffic_colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}
                traffic_emojis = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                
                metrics_html = f"""
                <div class="metric-container">
                    <div class="metric">
                        <strong>📏 Distance</strong><br>
                        {distance_km:.2f} km
                    </div>
                    <div class="metric">
                        <strong>🌤️ Weather</strong><br>
                        {weather_main}
                    </div>
                </div>
                <div class="metric-container">
                    <div class="metric">
                        <strong>🕒 Time</strong><br>
                        {time_of_day}
                    </div>
                    <div class="metric">
                        <strong>🚦 Traffic</strong><br>
                        <span style="color: {traffic_colors[traffic_level]}">
                            {traffic_emojis[traffic_level]} {traffic_level}
                        </span><br>
                        <small>{traffic_duration}</small>
                    </div>
                </div>
                """
                
                # Add calculated time display if available
                if calculated_time:
                    calculated_time_html = f"""
                    <div class="info-card" style="margin: 1rem 0; background: linear-gradient(135deg, #4CAF50, #45a049); color: white;">
                        <h4 style="margin: 0 0 0.5rem 0; color: white;">⏱️ Quick Estimate</h4>
                        <p style="margin: 0; font-size: 1.2rem;">
                            <strong>Travel Time:</strong> {normal_duration} + <strong>Prep Time:</strong> {prep_time} min = 
                            <strong style="font-size: 1.4rem;">{calculated_time} minutes total</strong>
                        </p>
                    </div>
                    """
                    st.markdown(calculated_time_html, unsafe_allow_html=True)
                st.markdown(metrics_html, unsafe_allow_html=True)
                
                # Make prediction
                try:
                    model = joblib.load("best_gradient_boosting_model.pkl")
                    
                    input_data = pd.DataFrame({
                        "Distance_km": [distance_km],
                        "Weather": [weather_main],
                        "Traffic_Level": [traffic_level],
                        "Time_of_Day": [time_of_day],
                        "Vehicle_Type": [vehicle_type],
                        "Preparation_Time_min": [prep_time],
                        "Courier_Experience_yrs": [experience]
                    })
                    
                    prediction = model.predict(input_data)[0]
                    
                    # Display prediction
                    prediction_html = f"""
                    <div class="prediction-card">
                        <h2>🚀 ML Model Prediction</h2>
                        <h1 style="font-size: 3rem; margin: 0.5rem 0;">{prediction:.0f}</h1>
                        <h3>minutes</h3>
                        <p>Using {vehicle_type} • {experience} years experience</p>
                    </div>
                    """
                    st.markdown(prediction_html, unsafe_allow_html=True)
                    
                    # Comparison between calculated and predicted time
                    if calculated_time:
                        diff = abs(prediction - calculated_time)
                        comparison_color = "#28a745" if diff <= 5 else "#ffc107" if diff <= 10 else "#dc3545"
                        
                        comparison_html = f"""
                        <div class="info-card" style="text-align: center; border-left: 4px solid {comparison_color};">
                            <h4>📊 Comparison</h4>
                            <p><strong>Quick Estimate:</strong> {calculated_time} min</p>
                            <p><strong>ML Prediction:</strong> {prediction:.0f} min</p>
                            <p style="color: {comparison_color}; font-weight: bold;">
                                Difference: {diff:.0f} minutes
                            </p>
                        </div>
                        """
                        st.markdown(comparison_html, unsafe_allow_html=True)
                    
                except FileNotFoundError:
                    st.error("🚫 Model file not found. Please ensure 'best_gradient_boosting_model.pkl' is available.")
                except Exception as e:
                    st.error(f"🚫 Prediction error: {str(e)}")
                    
            else:
                st.error("🚫 Failed to fetch distance data. Please try again.")
                
        except Exception as e:
            st.error(f"🚫 API Error: {str(e)}")
    else:
        st.markdown("### 👋 Welcome!")
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Get Started:</h4>
            <p>Click anywhere on the map to select your delivery location. 
            The app will automatically calculate delivery time based on:</p>
            <ul>
                <li>📍 Distance from restaurant</li>
                <li>🌤️ Current weather conditions</li>
                <li>🚦 Real-time traffic data</li>
                <li>⏰ Time of day</li>
                <li>🚗 Vehicle type & courier experience</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚴 Food Delivery Time Predictor • Made with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)