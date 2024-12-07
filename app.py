import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

# Streamlit Title
st.title("Wheat Price Prediction Dashboard")

# Custom CSS for UI styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f4f9;
            padding: 30px;
            border-radius: 10px;
        }
        .header {
            color: #4CAF50;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .subheader {
            color: #555;
            font-size: 22px;
            margin-top: 20px;
            text-align: center;
        }
        .table-container {
            margin-top: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .chart-container {
            margin-top: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding-bottom: 30px;
        }
        .st-selectbox {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        .st-button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Step 1: Connect to MySQL database
connection = mysql.connector.connect(
    host="localhost",         
    user="root",     
    password="PennePasta1224", 
    database="food_data"  
)

# Step 2: Write a SQL query
query = "SELECT * FROM food_data_table"

# Step 3: Use pandas to execute the query and temporarily store data
data = pd.read_sql(query, connection)  # Temporary storage in a DataFrame
connection.close()

# Perform operations on the DataFrame
data = data[data['commodity'] == 'Wheat']  # Filtering for Wheat
data = data.sort_values(by='date')  # Sorting by Date

# Creating lag and rolling mean features
data['lag_1'] = data.groupby(['state', 'district', 'market', 'commodity'])['price'].shift(1)
data['rolling_mean_3'] = data.groupby(['state', 'district', 'market', 'commodity'])['price'].transform(lambda x: x.rolling(window=3).mean())
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Drop rows with NaN values
data = data.dropna()

# One-Hot Encoding for categorical variables
encoded_data = pd.get_dummies(data, columns=['state', 'district', 'market', 'category', 'commodity'], drop_first=True)
encoded_data = encoded_data.drop(['date', 'unit', 'priceflag'], axis=1)

# Define features (X) and target (y)
X = encoded_data.drop(['price'], axis=1)
y = encoded_data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest and Linear Regression models
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)

# User input for months to predict
prediction_type = st.selectbox("Select Prediction Duration", ["Short Term (3 Months)", "Long Term (12 Months)"])

if prediction_type == "Short Term (3 Months)":
    months_to_predict = 3
else:
    months_to_predict = 12

# Get unique states
unique_states = data['state'].unique()

# Generate future date range
last_date = data['date'].max()
future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq='MS')[1:]

# Store future results
future_results = []

for state in unique_states:
    filtered_data_state = data[(data['state'] == state) & (data['commodity'] == 'Wheat')]
    last_row_state = filtered_data_state.iloc[-1]

    # Create future data for this state
    future_data = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'month': future_dates.month
    })

    future_data['state'] = state
    future_data['district'] = last_row_state['district']
    future_data['market'] = last_row_state['market']
    future_data['category'] = last_row_state['category']
    future_data['commodity'] = 'Wheat'
    last_price = last_row_state['price']
    future_data['lag_1'] = last_price
    future_data['rolling_mean_3'] = last_price
    future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
    future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)

    encoded_future_data = pd.get_dummies(future_data, columns=['state', 'district', 'market', 'category', 'commodity'], drop_first=True)
    encoded_future_data = encoded_future_data.drop(['date'], axis=1)
    encoded_future_data = encoded_future_data.reindex(columns=X.columns, fill_value=0)

    # Predict future prices using Random Forest
    rf_future_predictions = rf_model.predict(encoded_future_data)

    future_state_results = pd.DataFrame({
        'Date': future_dates,
        'State': state,
        'Predicted Prices (per kg)': rf_future_predictions
    })

    future_results.append(future_state_results)

# Combine all future predictions
final_future_results = pd.concat(future_results, ignore_index=True)

# Display future predictions in a bigger table
final_future_results['Month_Year'] = final_future_results['Date'].dt.strftime('%b %Y')

# Create a better layout with proper spacing
st.markdown('<div class="table-container">', unsafe_allow_html=True)

st.write("### Future Wheat Price Predictions:")
st.dataframe(final_future_results[['Month_Year', 'State', 'Predicted Prices (per kg)']], width=1500, height=600)

st.markdown('</div>', unsafe_allow_html=True)

# Plot the predicted prices for each state with a bigger chart size
st.markdown('<div class="chart-container">', unsafe_allow_html=True)

plt.figure(figsize=(20, 12))  # Increase the size of the plot

for state in final_future_results['State'].unique():
    state_data = final_future_results[final_future_results['State'] == state]
    plt.plot(state_data['Month_Year'], state_data['Predicted Prices (per kg)'], label=state)

xticks = final_future_results['Month_Year'].unique()
plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=45, ha='right')

plt.xlabel('Month Year', fontsize=16)
plt.ylabel('Predicted Prices (per kg)', fontsize=16)
plt.title('Predicted Wheat Prices for Different States', fontsize=18)
plt.legend(title='State', fontsize=14)

plt.tight_layout()
st.pyplot(plt)

st.markdown('</div>', unsafe_allow_html=True)
