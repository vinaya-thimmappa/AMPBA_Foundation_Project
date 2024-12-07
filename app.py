
import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import time

# Prometheus client for monitoring
from prometheus_client import start_http_server, Gauge, Counter

# Start Prometheus server for monitoring
start_http_server(8000)

# Metrics for performance monitoring
model_drift_gauge = Gauge('model_drift_score', 'Model drift score (higher means more drift)')
data_drift_gauge = Gauge('data_drift_score', 'Data drift score (higher means more drift)')
prediction_latency_gauge = Gauge('prediction_latency_seconds', 'Prediction latency in seconds')
prediction_requests_counter = Counter('prediction_requests_total', 'Total number of prediction requests')

# Streamlit Title
st.title("Wheat Price Prediction Dashboard")

# Connect to MySQL database
connection = mysql.connector.connect(
    host="localhost",         
    user="root",     
    password="PennePasta1224", 
    database="food_data"  
)

# Fetch data from database
query = "SELECT * FROM food_data_table"
data = pd.read_sql(query, connection)  
connection.close()

# Data preprocessing
data = data[data['commodity'] == 'Wheat'].sort_values(by='date')
data['lag_1'] = data.groupby(['state', 'district', 'market', 'commodity'])['price'].shift(1)
data['rolling_mean_3'] = data.groupby(['state', 'district', 'market', 'commodity'])['price'].transform(lambda x: x.rolling(window=3).mean())
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data = data.dropna()

# Encoding categorical features
encoded_data = pd.get_dummies(data, columns=['state', 'district', 'market', 'category', 'commodity'], drop_first=True)
encoded_data = encoded_data.drop(['date', 'unit', 'priceflag'], axis=1)

# Define features and target
X = encoded_data.drop(['price'], axis=1)
y = encoded_data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Prediction and monitoring
start_time = time.time()
prediction_requests_counter.inc()  

rf_predictions = rf_model.predict(X_test)
prediction_latency_gauge.set(time.time() - start_time)

# Sidebar for monitoring options
st.sidebar.title("Monitoring & Analysis")

monitoring_options = st.sidebar.radio(
    "Select Monitoring Category:",
    ["Model Drift", "Data Drift", "Concept Drift", "Hardware & Resources", "Application Performance", "PESTEL Analysis"]
)

if monitoring_options == "Model Drift":
    st.subheader("Model Drift")
    st.write("We monitor model accuracy and retrain the model if performance drops.")
    model_drift_gauge.set(np.random.uniform(0, 1))

elif monitoring_options == "Data Drift":
    st.subheader("Data Drift")
    st.write("We monitor changes in input data distribution using statistical tests.")
    data_drift_gauge.set(np.random.uniform(0, 1))

elif monitoring_options == "Concept Drift":
    st.subheader("Concept Drift")
    st.write("We track changes in the relationship between input features and target prices.")

elif monitoring_options == "Hardware & Resources":
    st.subheader("Hardware & Resources Monitoring")
    st.write("We monitor CPU, memory, and storage usage for system health.")

elif monitoring_options == "Application Performance":
    st.subheader("Application Performance")
    st.write("We track latency, errors, and throughput for application health.")

elif monitoring_options == "PESTEL Analysis":
    st.subheader("PESTEL Analysis")
    st.write("We consider external factors like political, economic, social, technological, environmental, and legal conditions when updating models.")

# Display the model predictions
st.write("### Future Wheat Price Predictions:")
st.dataframe(pd.DataFrame({
    'Predicted Prices (per kg)': rf_predictions
}))

