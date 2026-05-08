import gradio as gr
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("energy_model.pkl")
scaler = joblib.load("energy_scaler.pkl")

def predict_energy(lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4,
                   T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9,
                   T_out, Press_mm_hg, RH_out, Windspeed, Visibility,
                   Tdewpoint, rv1, rv2):
    features = np.array([[lights, T1, RH_1, T2, RH_2, T3, RH_3, T4, RH_4,
                          T5, RH_5, T6, RH_6, T7, RH_7, T8, RH_8, T9, RH_9,
                          T_out, Press_mm_hg, RH_out, Windspeed, Visibility,
                          Tdewpoint, rv1, rv2]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    return round(float(prediction), 2)

inputs = [
    gr.Slider(0, 100, value=10, label="Lights (Wh)"),
    gr.Slider(15, 30, value=20, step=0.1, label="T1 - Kitchen Temp (°C)"),
    gr.Slider(20, 80, value=45, step=0.1, label="RH_1 - Kitchen Humidity (%)"),
    gr.Slider(15, 30, value=19, step=0.1, label="T2 - Living Room Temp (°C)"),
    gr.Slider(20, 80, value=45, step=0.1, label="RH_2 - Living Room Humidity (%)"),
    gr.Slider(15, 30, value=21, step=0.1, label="T3 - Laundry Temp (°C)"),
    gr.Slider(20, 80, value=50, step=0.1, label="RH_3 - Laundry Humidity (%)"),
    gr.Slider(15, 30, value=21, step=0.1, label="T4 - Office Temp (°C)"),
    gr.Slider(20, 80, value=45, step=0.1, label="RH_4 - Office Humidity (%)"),
    gr.Slider(15, 30, value=19, step=0.1, label="T5 - Bathroom Temp (°C)"),
    gr.Slider(20, 80, value=50, step=0.1, label="RH_5 - Bathroom Humidity (%)"),
    gr.Slider(-10, 30, value=7, step=0.1, label="T6 - Outside Temp (north) (°C)"),
    gr.Slider(20, 100, value=80, step=0.1, label="RH_6 - Outside Humidity (north) (%)"),
    gr.Slider(15, 30, value=19, step=0.1, label="T7 - Ironing Temp (°C)"),
    gr.Slider(20, 80, value=45, step=0.1, label="RH_7 - Ironing Humidity (%)"),
    gr.Slider(15, 30, value=20, step=0.1, label="T8 - Teen Room Temp (°C)"),
    gr.Slider(20, 80, value=45, step=0.1, label="RH_8 - Teen Room Humidity (%)"),
    gr.Slider(15, 30, value=20, step=0.1, label="T9 - Parent Room Temp (°C)"),
    gr.Slider(20, 80, value=45, step=0.1, label="RH_9 - Parent Room Humidity (%)"),
    gr.Slider(-10, 30, value=6, step=0.1, label="T_out - Outside Temp (°C)"),
    gr.Slider(700, 800, value=733, step=0.1, label="Pressure (mm Hg)"),
    gr.Slider(20, 100, value=75, step=0.1, label="RH_out - Outside Humidity (%)"),
    gr.Slider(0, 20, value=5, step=0.1, label="Windspeed (m/s)"),
    gr.Slider(0, 100, value=40, step=0.1, label="Visibility (km)"),
    gr.Slider(-10, 20, value=3, step=0.1, label="Tdewpoint (°C)"),
    gr.Slider(0, 100, value=50, step=0.1, label="rv1 (random variable 1)"),
    gr.Slider(0, 100, value=50, step=0.1, label="rv2 (random variable 2)"),
]

demo = gr.Interface(
    fn=predict_energy,
    inputs=inputs,
    outputs=gr.Number(label="Predicted Appliance Energy Consumption (Wh)"),
    title="🔋 Household Appliance Energy Consumption Predictor",
    description="Predict the energy consumption of household appliances based on temperature, humidity, and other environmental sensor data. Adjust the sliders and click Submit to get a prediction.",
    theme=gr.themes.Soft(),
)

demo.launch()