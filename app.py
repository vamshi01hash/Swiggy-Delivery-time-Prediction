import streamlit as st
import pandas as pd
import joblib 

st.title("Swiggy Delivery Time Prediction")

df = pd.read_csv('swiggy_cleaned.csv')
df.dropna(inplace=True)
X = df.copy()

model = joblib.load("model.pkl")

col = [
    'age', 'ratings', 'weather', 'traffic', 'vehicle_condition',
    'type_of_order', 'type_of_vehicle', 'multiple_deliveries', 'festival',
    'city_type', 'city_name', 'order_day_of_week', 'is_weekend',
    'pickup_time_minutes', 'order_time_hour', 'order_time_of_day',
    'distance'
]

Age = st.number_input('Enter The Age:', min_value=1, max_value=50)
Rating = st.number_input('Enter The Rating:', min_value=1, max_value=5)
weather = st.selectbox('Weather', X['weather'].unique())
traffic = st.selectbox('Traffic', X['traffic'].unique())
vehicle_condition = st.selectbox('Vehicle Condition', X['vehicle_condition'].unique())
type_of_order = st.selectbox('Type of Order', X['type_of_order'].unique())
type_of_vehicle = st.selectbox('Type of Vehicle', X['type_of_vehicle'].unique())
multiple_deliveries = st.number_input('Number of Deliveries:', min_value=1, max_value=3)
festival = st.selectbox('Festival', X['festival'].unique())
city_type = st.selectbox('City Type', X['city_type'].unique())
city_name = st.selectbox('City Name', X['city_name'].unique())
order_day_of_week = st.selectbox('Order Day of Week', X['order_day_of_week'].unique())
is_weekend = st.number_input('Is it Weekend (0/1):', min_value=0, max_value=1)
pickup_time_minutes = st.number_input('Pickup Time Minutes:', min_value=1, max_value=100)
order_time_hour = st.number_input('Order Time Hour:', min_value=0, max_value=23)
order_time_of_day = st.selectbox('Order Time of Day:', X['order_time_of_day'].unique())
distance = st.number_input('Distance:', min_value=1, max_value=100)

data = pd.DataFrame([[
    Age, Rating, weather, traffic, vehicle_condition,
    type_of_order, type_of_vehicle, multiple_deliveries, festival,
    city_type, city_name, order_day_of_week, is_weekend,
    pickup_time_minutes, order_time_hour, order_time_of_day, distance
]], columns=col)

if st.button('Predict'):
    result = model.predict(data)[0]
    st.success(f"Estimated Delivery Time: {result} minutes")
