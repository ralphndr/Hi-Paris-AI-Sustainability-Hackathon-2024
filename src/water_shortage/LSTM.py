import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Example time series dataset
dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
values = np.sin(np.arange(1000) * 2 * np.pi / 365) + np.random.normal(0, 0.1, size=1000)
df = pd.DataFrame({"timestamp": dates, "value": values})

# Ensure 'timestamp' is a datetime object and sort by it
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(by='timestamp', inplace=True)

# Set 'timestamp' as the index
df.set_index('timestamp', inplace=True)

# Visualize the dataset
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['value'])
plt.title("Time Series Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()

# Preprocessing
def prepare_time_series_data(df, feature_col, lookback):
    """
    Prepares time series data for supervised learning using a lookback window.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a DateTime index.
    - feature_col (str): Name of the column containing the feature values.
    - lookback (int): Number of past observations to use for predicting the next value.
    
    Returns:
    - X, y: Prepared features and labels as NumPy arrays.
    - scaler: The fitted MinMaxScaler instance for inverse transformations.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature_col]])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])  # Use the last `lookback` observations
        y.append(scaled_data[i, 0])  # Predict the next observation
    
    return np.array(X), np.array(y), scaler

# Parameters
lookback = 30  # Use the past 30 days to predict the next value
X, y, scaler = prepare_time_series_data(df, feature_col='value', lookback=lookback)

# Split the data into training and testing sets
split_idx = int(len(X) * 0.8)  # 80% training, 20% testing
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for LSTM input (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Predict a single output (next value)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")

# Predict the test data
y_pred = model.predict(X_test)
y_pred.to_csv("D:/HiParis_Hackathon/y_test_LSTM.csv", index=False)

# Reverse scaling for visualization
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_rescaled, label="Actual")
plt.plot(df.index[-len(y_test):], y_pred_rescaled, label="Predicted", linestyle="dashed")
plt.title("Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
