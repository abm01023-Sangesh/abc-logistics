
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input features (matching X.columns)
delivery_distance = st.slider('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 is high)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-5, 5 is severe)', min_value=1, max_value=5, value=3)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
driver_experience = st.slider('Driver Experience (years)', min_value=0, max_value=20, value=5)
num_stops = st.slider('Number of Stops', min_value=1, max_value=10, value=5)
vehicle_age = st.slider('Vehicle Age (years)', min_value=0, max_value=15, value=5)
road_condition_score = st.slider('Road Condition Score (1-5, 5 is good)', min_value=1, max_value=5, value=3)
package_weight = st.slider('Package Weight (kg)', min_value=0.0, max_value=50.0, value=10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', min_value=5.0, max_value=25.0, value=15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', min_value=0, max_value=120, value=60)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f'The delivery is predicted to be **DELAYED** (Probability: {prediction_proba[1]:.2f})')
    else:
        st.success(f'The delivery is predicted to be **ON TIME** (Probability: {prediction_proba[0]:.2f})')

    st.write('---')
    st.subheader('Input Features:')
    st.write(input_data)

