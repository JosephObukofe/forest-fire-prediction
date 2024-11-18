import requests
from predicting_forest_fires.config.config import FAST_API_BASE_URL

SUBMIT_INPUT_URL = f"{FAST_API_BASE_URL}/submit_input/"

#SUBMIT_INPUT_URL = "http://localhost:8000/submit_input/"  # Adjust if necessary

input_data = {
    "X": 1,
    "Y": 1,
    "month": "mar",
    "day": "fri",
    "DMC": 45.3,
    "FFMC": 85.1,
    "DC": 100.6,
    "ISI": 4.5,
    "temp": 22.1,
    "RH": 45,
    "wind": 2.5,
    "rain": 0.0
}

try:
    response = requests.post(SUBMIT_INPUT_URL, json={"features": input_data})
    if response.status_code == 200:
        print("POST request successful:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Error occurred: {e}")
