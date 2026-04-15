import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# ADD YOUR OPENAI KEY
# -----------------------------

client = OpenAI(api_key="")

st.set_page_config(page_title="Smart Irrigation AI", layout="wide")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Smart Irrigation AI")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Irrigation Prediction",
        "Crop Recommendation",
        "Weather & Rainfall",
        "Voice AI Assistant"
    ]
)

# -----------------------------
# LOAD DATA
# -----------------------------

irrigation_df = pd.read_csv("irrigation_data.csv")
crop_df = pd.read_csv("crop_dataset.csv")

# -----------------------------
# TRAIN IRRIGATION MODEL
# -----------------------------

features = ["Soil_Moisture","Temperature","Humidity","Rainfall"]

X = irrigation_df[features]
y = irrigation_df["Irrigation"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()
model.fit(X_train,y_train)

# -----------------------------
# TRAIN CROP MODEL
# -----------------------------

crop_features = ["N","P","K","temperature","humidity","ph","rainfall"]

Xc = crop_df[crop_features]
yc = crop_df["label"]

X_train_c,X_test_c,y_train_c,y_test_c = train_test_split(
    Xc,yc,test_size=0.2,random_state=42
)

crop_model = RandomForestClassifier()
crop_model.fit(X_train_c,y_train_c)

# -----------------------------
# DASHBOARD
# -----------------------------

if menu == "Dashboard":

    st.title("Smart Agriculture Dashboard")

    lat = st.number_input("Latitude", value=28.61)
    lon = st.number_input("Longitude", value=77.20)

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    try:
        weather = requests.get(url).json()
        temperature = weather["current_weather"]["temperature"]
        wind = weather["current_weather"]["windspeed"]
    except:
        temperature = 0
        wind = 0

    soil = st.slider("Soil Moisture",0,100,40)
    humidity = st.slider("Humidity",0,100,60)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Temperature", str(temperature) + " °C")
    col2.metric("Wind Speed", str(wind) + " km/h")
    col3.metric("Soil Moisture", str(soil) + " %")
    col4.metric("Humidity", str(humidity) + " %")

    st.write("AI Irrigation Score")

    irrigation_score = 100 - soil
    st.progress(irrigation_score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=100-soil,
        title={'text': "Drought Risk %"},
        gauge={'axis': {'range':[0,100]}}
    ))

    st.plotly_chart(fig)

# -----------------------------
# IRRIGATION PREDICTION
# -----------------------------

elif menu == "Irrigation Prediction":

    st.header("Irrigation Prediction")

    soil = st.slider("Soil Moisture",0,100,30)
    temp = st.slider("Temperature",0,50,30)
    hum = st.slider("Humidity",0,100,50)
    rain = st.slider("Rainfall",0,50,5)

    if st.button("Predict Irrigation"):

        input_data = pd.DataFrame(
            [[soil,temp,hum,rain]],
            columns=features
        )

        prediction = model.predict(input_data)

        if prediction[0]==1:
            st.success("Irrigation Needed")
        else:
            st.info("No Irrigation Needed")

# -----------------------------
# CROP RECOMMENDATION
# -----------------------------

elif menu == "Crop Recommendation":

    st.header("Crop Recommendation")

    N = st.slider("Nitrogen",0,140,50)
    P = st.slider("Phosphorus",0,140,40)
    K = st.slider("Potassium",0,140,40)

    temp2 = st.slider("Temperature",0,40,25)
    humidity2 = st.slider("Humidity",0,100,60)
    ph = st.slider("Soil pH",0.0,14.0,6.5)
    rainfall2 = st.slider("Rainfall",0,300,100)

    if st.button("Recommend Crop"):

        crop_input = pd.DataFrame(
            [[N,P,K,temp2,humidity2,ph,rainfall2]],
            columns=crop_features
        )

        prediction = crop_model.predict(crop_input)

        st.success("Recommended Crop: " + prediction[0])

# -----------------------------
# WEATHER
# -----------------------------

elif menu == "Weather & Rainfall":

    st.header("Live Weather")

    lat = st.number_input("Latitude", value=28.61)
    lon = st.number_input("Longitude", value=77.20)

    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    try:

        weather=requests.get(url).json()

        temp_live=weather["current_weather"]["temperature"]
        wind=weather["current_weather"]["windspeed"]

        st.metric("Temperature",str(temp_live)+" °C")
        st.metric("Wind Speed",str(wind)+" km/h")

    except:

        st.warning("Weather API unavailable")

# -----------------------------
# VOICE AI ASSISTANT
# -----------------------------

elif menu == "Voice AI Assistant":

    st.header("Voice AI Farming Assistant")

    if st.button("Start Voice Assistant"):

        recognizer = sr.Recognizer()

        with sr.Microphone() as source:

            st.info("Listening... Speak now")

            audio = recognizer.listen(source)

        try:

            text = recognizer.recognize_google(audio)

            st.success("You said: " + text)
            openai.api_key = "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"


            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are an agriculture expert helping farmers."},
                    {"role":"user","content":text}
                ]
            )

            answer = response.choices[0].message.content

            st.success(answer)

            tts = gTTS(answer)

            tts.save("response.mp3")

            st.audio("response.mp3")

        except Exception as e:

            st.error(str(e))