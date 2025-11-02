import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import json
import os
from datetime import datetime
import requests

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="LungGuard AI™",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS – Cyberpunk 2077 Vibes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #00ffea;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background: transparent;
    }
    .css-1d391kg, .css-1v0mbdj { color: #00ffea; }
    .stButton>button {
        background: linear-gradient(45deg, #ff006e, #8338ec);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(255,0,110,0.6);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(255,0,110,0.9);
    }
    .glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        margin: 10px 0;
    }
    .pulse-red { animation: pulse 1.5s infinite; }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,0,0,0.7); }
        70% { box-shadow: 0 0 0 20px rgba(255,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,0,0,0); }
    }
</style>
""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return joblib.load('Endpoint.pkl')

model = load_model()

# ------------------- 3D LUNG MODEL -------------------
def create_3d_lung(probability):
    # Simplified lung geometry (left + right lobes)
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(-1, 1, 50)
    theta, z = np.meshgrid(theta, z)
    
    # Left lung
    x_left = 0.4 * np.cos(theta) * (1 - 0.3 * np.abs(z))
    y_left = 0.6 * np.sin(theta) * (1 - 0.3 * np.abs(z)) - 0.4
    # Right lung
    x_right = 0.4 * np.cos(theta) * (1 - 0.3 * np.abs(z))
    y_right = 0.6 * np.sin(theta) * (1 - 0.3 * np.abs(z)) + 0.4

    color = 'cyan' if probability < 0.5 else 'red'
    opacity = 0.7 + 0.3 * probability
    intensity = int(255 * probability)

    fig = go.Figure()

    # Left lung
    fig.add_trace(go.Surface(
        x=x_left, y=y_left, z=z,
        colorscale=[[0, 'cyan'], [1, f'rgb({intensity},0,{255-intensity})']],
        opacity=opacity,
        showscale=False,
        name="Left Lung"
    ))

    # Right lung
    fig.add_trace(go.Surface(
        x=x_right, y=y_right, z=z,
        colorscale=[[0, 'cyan'], [1, f'rgb({intensity},0,{255-intensity})']],
        opacity=opacity,
        showscale=False,
        name="Right Lung"
    ))

    fig.update_layout(
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return fig

# ------------------- RISK RADAR CHART -------------------
def create_radar_chart(data):
    categories = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#ff006e',
        fillcolor='rgba(255,0,110,0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

# ------------------- VOICE INPUT -------------------
def voice_input():
    st.markdown("""
    <button onclick="startDictation()" style="padding:10px 20px; background:#ff006e; color:white; border:none; border-radius:50px;">
        🎤 Speak Symptoms
    </button>
    <script>
    function startDictation() {
        if (window.hasOwnProperty('webkitSpeechRecognition')) {
            var recognition = new webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            recognition.start();
            recognition.onresult = function(e) {
                document.getElementById('transcript').value = e.results[0][0].transcript;
                recognition.stop();
            };
            recognition.onerror = function(e) { recognition.stop(); }
        }
    }
    </script>
    <input id="transcript" placeholder="Voice input will appear here..." style="width:100%; padding:10px; margin-top:10px; border-radius:10px; border:1px solid #00ffea;">
    """, unsafe_allow_html=True)

# ------------------- MAIN APP -------------------
def main():
    st.markdown("<h1 style='text-align:center; color:#00ffea; text-shadow: 0 0 20px #00ffea;'>🫁 LungGuard AI™</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa;'>Next-Gen AI Lung Cancer Risk Engine</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.subheader("Patient Scan")

        gender = st.selectbox("Gender", ("Male", "Female"), key="gender")
        age = st.slider("Age", 18, 100, 40, key="age")
        
        symptoms = [
            "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure",
            "Chronic Disease", "Fatigue", "Allergy", "Wheezing",
            "Alcohol Consuming", "Coughing", "Shortness of Breath",
            "Swallowing Difficulty", "Chest Pain"
        ]
        inputs = {}
        for s in symptoms:
            inputs[s] = st.selectbox(s, ("No", "Yes"), key=s)

        st.markdown("</div>", unsafe_allow_html=True)

        # Voice input
        st.markdown("### 🎤 Or Speak Your Symptoms")
        voice_input()

    with col2:
        if st.button("SCAN NOW", key="scan"):
            with st.spinner("Analyzing lung tissue..."):
                # Convert inputs
                def to_num(val): return 1 if val in ["Yes", "Male"] else 0
                input_data = np.array([[
                    to_num(gender), age,
                    *[to_num(inputs[s]) for s in symptoms]
                ]])

                # Predict
                pred = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else pred

                # 3D Lung
                lung_fig = create_3d_lung(prob)
                st.plotly_chart(lung_fig, use_container_width=True, config={'displayModeBar': False})

                # Results
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Risk Level", f"{'HIGH' if pred else 'LOW'}", delta=f"{prob:.1%}")
                with col_b:
                    st.metric("Confidence", f"{max(model.predict_proba(input_data)[0]):.1%}")
                with col_c:
                    st.metric("Age Factor", f"{age} yrs")

                if pred:
                    st.markdown("<div class='pulse-red glass'>", unsafe_allow_html=True)
                    st.error("**LUNG CANCER DETECTED**")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.audio("https://www.soundjay.com/buttons/beep-07.mp3", format="audio/mp3", start_time=0)
                else:
                    st.success("**CLEAR – NO CANCER**")
                    st.balloons()
                    st.audio("https://www.soundjay.com/buttons/success-1.mp3", format="audio/mp3")

                # Radar Chart
                radar_data = {k: to_num(v)/1 for k, v in inputs.items()}
                radar_data["Age"] = (age - 18) / 82
                radar_data["Gender"] = to_num(gender)
                st.plotly_chart(create_radar_chart(radar_data), use_container_width=True)

                # Save to history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "risk": f"{prob:.1%}",
                    "result": "Cancer" if pred else "Clear"
                })

    # History
    if st.session_state.get("history"):
        st.sidebar.markdown("### History")
        for h in st.session_state.history[-5:]:
            st.sidebar.markdown(f"**{h['time']}** → {h['result']} ({h['risk']})")

    # Export
    if st.button("Export Report (PDF)"):
        st.info("PDF export coming soon! (Use browser print → Save as PDF)")

if __name__ == "__main__":
    main()