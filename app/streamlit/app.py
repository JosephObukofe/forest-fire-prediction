import requests
import websocket
import threading
import streamlit as st
from predicting_forest_fires.config.config import (
    FAST_API_BASE_URL,
    WEB_SOCKET_URL,
)

SUBMIT_INPUT_URL = f"{FAST_API_BASE_URL}/submit_input/"

st.set_page_config(page_title="Forest Fire Prediction", page_icon="🌲🔥", layout="wide")
st.title("Forest Fire Prediction 🌲🔥")
st.markdown(
    "Welcome! Enter values for the input features below to predict the likelihood of a forest fire."
)

with st.sidebar:
    st.header("About the App")
    st.info(
        "💡 This application uses machine learning to predict forest fires based on environmental data. "
        "Enter your inputs on the left and submit them to receive predictions."
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Additional Resources 📚")
    st.write(
        "[Model Documentation](https://github.com/JosephObukofe/forest-fire-prediction) | [Data Source](https://github.com/JosephObukofe/forest-fire-prediction)"
    )
    st.markdown("### Contact ✉️")
    st.write("[josephobukofe@gmail.com](mailto:josephobukofe@gmail.com)")
    st.markdown("<br>", unsafe_allow_html=True)

if "X" not in st.session_state:
    st.session_state.X = None
    st.session_state.Y = None
    st.session_state.month = "Select a month"
    st.session_state.day = "Select a day"
    st.session_state.DMC = None
    st.session_state.FFMC = None
    st.session_state.DC = None
    st.session_state.ISI = None
    st.session_state.temp = None
    st.session_state.RH = None
    st.session_state.wind = None
    st.session_state.rain = None


def create_spatial_datetime_inputs():
    st.subheader("Spatial & Date Information")
    month_options = [
        "Select a month",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    day_options = ["Select a day", "mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    month_index = (
        month_options.index(st.session_state.month)
        if st.session_state.month in month_options
        else 0
    )
    day_index = (
        day_options.index(st.session_state.day)
        if st.session_state.day in day_options
        else 0
    )

    return {
        "X": st.number_input(
            "X-axis Spatial Coordinate",
            min_value=1,
            max_value=9,
            value=st.session_state.X,
            key="X",
            help="X-axis spatial coordinate. Must be an integer between 1 and 9.",
        ),
        "Y": st.number_input(
            "Y-axis Spatial Coordinate",
            min_value=2,
            max_value=9,
            value=st.session_state.Y,
            key="Y",
            help="Y-axis spatial coordinate. Must be an integer between 2 and 9.",
        ),
        "month": st.selectbox(
            "Month",
            options=month_options,
            index=month_index,
            key="month",
            help="Month of the year. Choose one from 'jan', 'feb', 'mar', etc.",
        ),
        "day": st.selectbox(
            "Day",
            options=day_options,
            index=day_index,
            key="day",
            help="Day of the week. Choose one from 'mon', 'tue', 'wed', etc.",
        ),
    }


def create_moisture_inputs():
    st.subheader("Moisture Codes")
    return {
        "DMC": st.number_input(
            "Duff Moisture Code (DMC)",
            min_value=1.1,
            max_value=291.3,
            value=st.session_state.DMC,
            key="DMC",
            help="Duff Moisture Code indicating moisture content of deep organic layers. Must be between 1.1 and 291.3.",
        ),
        "FFMC": st.number_input(
            "Fine Fuel Moisture Code (FFMC)",
            min_value=18.7,
            max_value=96.2,
            value=st.session_state.FFMC,
            key="FFMC",
            help="Fine Fuel Moisture Code indicating moisture of litter and fine fuels. Must be between 18.7 and 96.2.",
        ),
        "DC": st.number_input(
            "Drought Code (DC)",
            min_value=7.9,
            max_value=860.6,
            value=st.session_state.DC,
            key="DC",
            help="Drought Code indicating moisture in compact organic layers. Must be between 7.9 and 860.6.",
        ),
        "ISI": st.number_input(
            "Initial Spread Index (ISI)",
            min_value=0.0,
            max_value=56.1,
            value=st.session_state.ISI,
            key="ISI",
            help="Initial Spread Index combining wind and FFMC for fire spread rate. Must be between 0.0 and 56.1.",
        ),
    }


def create_environmental_inputs():
    st.subheader("Environmental Conditions")
    return {
        "temp": st.number_input(
            "Temperature (°C)",
            min_value=2.2,
            max_value=33.3,
            value=st.session_state.temp,
            key="temp",
            help="Temperature recorded at noon (standard time). Must be between 2.2 and 33.3 °C.",
        ),
        "RH": st.number_input(
            "Relative Humidity (%)",
            min_value=15,
            max_value=100,
            value=st.session_state.RH,
            key="RH",
            help="Relative humidity recorded at noon (standard time). Must be between '15%' and '100%'.",
        ),
        "wind": st.number_input(
            "Wind Speed (km/h)",
            min_value=0.4,
            max_value=9.4,
            value=st.session_state.wind,
            key="wind",
            help="Wind speed recorded at noon (standard time). Must be between 0.4 and 9.4 km/h.",
        ),
        "rain": st.number_input(
            "Rain (mm/m)",
            min_value=0.0,
            max_value=6.4,
            value=st.session_state.rain,
            key="rain",
            help="Outside rain in mm/m recorded at noon (standard time). Must be between 0.0 and 6.4.",
        ),
    }


if st.button(
    "Clear Inputs 🧹",
    help="Reset all the inputs to default values",
    key="clear_inputs",
):
    st.session_state.X = None
    st.session_state.Y = None
    st.session_state.month = "Select a month"
    st.session_state.day = "Select a day"
    st.session_state.DMC = None
    st.session_state.FFMC = None
    st.session_state.DC = None
    st.session_state.ISI = None
    st.session_state.temp = None
    st.session_state.RH = None
    st.session_state.wind = None
    st.session_state.rain = None

st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    input_data1 = create_spatial_datetime_inputs()
with col2:
    input_data2 = create_moisture_inputs()
with col3:
    input_data3 = create_environmental_inputs()

input_data = {**input_data1, **input_data2, **input_data3}

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Submit for Prediction 🚀")
if st.button(
    "Submit 📩",
    help="Click to submit the input data for prediction",
    key="submit_btn",
):
    with st.spinner("Submitting data for prediction..."):
        st.progress(50)
        try:
            response = requests.post(SUBMIT_INPUT_URL, json={"features": input_data})
            if response.status_code == 200:
                task_info = response.json()
                st.success(
                    f"Prediction submitted successfully! Task ID: {task_info['task_id']}",
                    icon="✅",
                )
                st.write("Waiting for prediction results...")
            else:
                st.error(
                    f"Error submitting prediction: {response.status_code} - {response.text}",
                    icon="❌",
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")


def handle_prediction_result(message):
    try:
        prediction = message.strip().lower()
        st.write("### Prediction Result 🔮")
        prediction_box = st.container()
        with prediction_box:
            if prediction == "yes":
                st.success("🔥 A forest fire is likely! Here are your next steps:")
                st.markdown(
                    """
                    - **Notify authorities**: Contact local fire services immediately.
                    - **Evacuate the area**: If you're in the vicinity, ensure everyone is evacuated safely.
                    - **Monitor updates**: Keep an eye on official channels for further information.
                    """
                )
            elif prediction == "no":
                st.success("🌲 No forest fire is predicted. Here are your next steps:")
                st.markdown(
                    """
                    - **Maintain vigilance**: Stay aware of the environmental conditions.
                    - **Preventative measures**: Consider implementing fire prevention strategies, such as clearing dry brush.
                    - **Report changes**: If conditions worsen, report any signs of fire risk to the authorities.
                    """
                )
            else:
                st.warning("⚠️ No prediction available yet. Please wait.")
    except Exception as e:
        st.error(f"Error processing prediction result: {e}")


def listen_for_predictions():
    ws = websocket.WebSocketApp(
        WEB_SOCKET_URL,
        on_message=lambda _, message: handle_prediction_result(message),
        on_error=lambda _, error: st.error(f"WebSocket error: {error}"),
        on_close=lambda _: st.write("WebSocket connection closed."),
    )
    ws.run_forever()


if "ws_thread_started" not in st.session_state:
    st.session_state.ws_thread_started = True
    thread = threading.Thread(target=listen_for_predictions)
    thread.daemon = True
    thread.start()

st.markdown("---")
st.write(
    "🔍 **Want to learn more?** [Explore the project documentation here](https://github.com/JosephObukofe/forest-fire-prediction)."
)
