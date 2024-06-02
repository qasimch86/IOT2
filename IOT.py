#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

def generate_sensor_data():
    temperature = round(random.uniform(20, 30), 2)  # Simulate temperature readings between 20°C and 30°C
    humidity = round(random.uniform(40, 60), 2)     # Simulate humidity readings between 40% and 60%
    pressure = round(random.uniform(900, 1100), 2)  # Simulate pressure readings between 900 hPa and 1100 hPa
    return temperature, humidity, pressure

# Example usage:
temperature, humidity, pressure = generate_sensor_data()
print("Temperature:", temperature, "°C")
print("Humidity:", humidity, "%")
print("Pressure:", pressure, "hPa")


# In[2]:


import csv

def collect_sensor_data(num_samples):
    with open('sensor_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)'])
        for _ in range(num_samples):
            temperature, humidity, pressure = generate_sensor_data()
            writer.writerow([temperature, humidity, pressure])

# Example usage:
collect_sensor_data(100)  # Collect 100 samples of sensor data


# In[3]:


import csv

def collect_sensor_data(num_samples):
    with open('sensor_data.csv', 'w', newline='', encoding='utf-8') as csvfile:  # Specify encoding here
        writer = csv.writer(csvfile)
        writer.writerow(['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)'])
        for _ in range(num_samples):
            temperature, humidity, pressure = generate_sensor_data()
            writer.writerow([temperature, humidity, pressure])

# Example usage:
collect_sensor_data(100)  # Collect 100 samples of sensor data


# In[4]:


import pandas as pd

# Load data from CSV file into a DataFrame
sensor_data = pd.read_csv('sensor_data.csv')

# Basic statistics
print("Mean:")
print(sensor_data.mean())
print("\nMedian:")
print(sensor_data.median())
print("\nStandard Deviation:")
print(sensor_data.std())

# Anomaly detection (example: detect temperatures above 25°C)
anomalies = sensor_data[sensor_data['Temperature (°C)'] > 25]
print("\nAnomalies (Temperature > 25°C):")
print(anomalies)


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histograms
plt.figure(figsize=(10, 6))
sns.histplot(sensor_data['Temperature (°C)'], kde=True, color='blue', bins=20)
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot (Temperature vs Humidity)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=sensor_data, x='Temperature (°C)', y='Humidity (%)', color='green')
plt.title('Temperature vs Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.show()


# In[8]:


import dash
from dash import dcc, html
import dash.dependencies as dd
import plotly.graph_objs as go
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout
app.layout = html.Div([
    html.H1('IoT Sensor Data Dashboard'),
    dcc.Graph(id='temperature-graph'),
    dcc.Graph(id='humidity-graph'),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds (60 seconds)
        n_intervals=0
    )
])

# Callbacks
@app.callback(
    [dd.Output('temperature-graph', 'figure'),
     dd.Output('humidity-graph', 'figure')],
    [dd.Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Retrieve data
    sensor_data = pd.read_csv('sensor_data.csv')
    
    # Ensure the index is datetime if applicable
    # sensor_data.index = pd.to_datetime(sensor_data['Time'])  # Uncomment and modify if there's a 'Time' column

    # Create temperature graph
    temperature_trace = go.Scatter(x=sensor_data.index, y=sensor_data['Temperature (°C)'],
                                   mode='lines', name='Temperature (°C)')
    temperature_layout = go.Layout(title='Temperature Over Time', xaxis=dict(title='Time'), yaxis=dict(title='Temperature (°C)'))
    temperature_fig = go.Figure(data=[temperature_trace], layout=temperature_layout)
    
    # Create humidity graph
    humidity_trace = go.Scatter(x=sensor_data.index, y=sensor_data['Humidity (%)'],
                                mode='lines', name='Humidity (%)', marker=dict(color='green'))
    humidity_layout = go.Layout(title='Humidity Over Time', xaxis=dict(title='Time'), yaxis=dict(title='Humidity (%)'))
    humidity_fig = go.Figure(data=[humidity_trace], layout=humidity_layout)
    
    return temperature_fig, humidity_fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)


# In[9]:


pip install pandas statsmodels matplotlib


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Load sensor data
sensor_data = pd.read_csv('sensor_data.csv')

# Fit ARIMA model (example for temperature data)
def arima_forecast(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    fitted_model = model.fit(disp=0)
    forecast, stderr, conf_int = fitted_model.forecast(steps=10)  # Forecasting next 10 steps
    return forecast, conf_int

# Example on Temperature data
temperature_data = sensor_data['Temperature (°C)']
forecast, conf_int = arima_forecast(temperature_data)

# Plotting the forecast
plt.figure(figsize=(10, 5))
plt.plot(temperature_data, label='Observed')
plt.plot(range(len(temperature_data), len(temperature_data) + 10), forecast, label='Forecast')
plt.fill_between(range(len(temperature_data), len(temperature_data) + 10),
                 conf_int[:, 0], conf_int[:, 1], color='k', alpha=.15)
plt.legend()
plt.title('Temperature Forecast using ARIMA')
plt.show()


# In[12]:


pip install pandas numpy tensorflow scikit-learn matplotlib


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load sensor data
sensor_data = pd.read_csv('sensor_data.csv')
temperature_data = sensor_data['Temperature (°C)'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(temperature_data)

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, Y_train, batch_size=1, epochs=1)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(scaled_data), label='Observed')
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='Train Predict')
plt.plot(np.arange(time_step + len(train_predict) + 2, time_step + len(train_predict) + 2 + len(test_predict)), test_predict, label='Test Predict')
plt.legend()
plt.title('Temperature Forecast using LSTM')
plt.show()


# In[ ]:




