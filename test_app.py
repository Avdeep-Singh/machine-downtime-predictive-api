# test_app.py
from fastapi.testclient import TestClient
from app import app
import pandas as pd
import numpy as np

client = TestClient(app)

# Example test data 
test_data = {
    "Machine_ID": "Makino-L1-Unit1-2013",
    "Assembly_Line_No": "Shopfloor-L1",
    "Hydraulic_Pressure(bar)": 71.04,
    "Coolant_Pressure(bar)": 6.933724915,
    "Air_System_Pressure(bar)": 6.284964506,
    "Coolant_Temperature": 25.6,
    "Hydraulic_Oil_Temperature(?C)": 46.0,
    "Spindle_Bearing_Temperature(?C)": 33.4,
    "Spindle_Vibration(?m)": 1.291,
    "Tool_Vibration(?m)": 26.492,
    "Spindle_Speed(RPM)": 25892.0,
    "Voltage(volts)": 335.0,
    "Torque(Nm)": 24.05532601,
    "Cutting(kN)": 3.58
}


def test_upload():
    with open("data/uploaded_data.csv", "rb") as f:
        response = client.post("/upload", files={"file": f})
    assert response.status_code == 200
    assert "Data uploaded successfully" in response.json()["message"]

def test_train():
    response = client.post("/train")
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "f1_score" in response.json()

def test_predict():
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "Downtime" in response.json()
    assert "Confidence" in response.json()

# Test for invalid input
def test_predict_invalid_input():
    invalid_data = {"Temperature": "hot", "Run_Time": 120}  # Incorrect data type
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 500