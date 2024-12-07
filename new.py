import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# App Configuration
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# Add a info
st.markdown("""
# 🚨 Disclaimer 🚨
**This is not financial advice.**  
""", unsafe_allow_html=True)

st.title("📈 Stock Dashboard")

default_tickers = ["AAPL", "MSFT", "GOOGL"]  # List of default tickers

st.header("Default Stock Overview with Daily Charts")

fig = make_subplots(rows=len(default_tickers), cols=1, shared_xaxes=True, subplot_titles=default_tickers)

for i, default_ticker in enumerate(default_tickers, start=1):
    try:
        stock = yf.Ticker(default_ticker)
        info = stock.info

        # Display metrics
        st.subheader(f"{info.get('shortName', 'N/A')} ({default_ticker})")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            st.metric("Day High", f"${info.get('dayHigh', 'N/A')}")
            st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        with col2:
            st.metric("Open", f"${info.get('open', 'N/A')}")
            st.metric("Day Low", f"${info.get('dayLow', 'N/A')}")
            st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

        # Fetch and add daily historical data to the subplot
        hist = stock.history(period="5d")  # Fetch last 5 days for small chart
        fig.add_trace(
            go.Scatter(
                x=hist.index, 
                y=hist['Close'], 
                mode='lines+markers', 
                name=f"{default_ticker}",
                showlegend=False
            ),
            row=i,
            col=1
        )

    except Exception as e:
        st.warning(f"Could not fetch data for {default_ticker}: {e}")

# Update layout of the combined figure
fig.update_layout(height=300 * len(default_tickers), title_text="Daily Charts for Default Tickers", showlegend=False)
st.plotly_chart(fig)

# Search Bar for User-Input Ticker
st.header("Search Stock by Ticker")
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, AAPL):", "")

if ticker:
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Display Stock Info
        st.header(f"{info.get('shortName', 'N/A')} ({ticker.upper()})")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            st.metric("Day High", f"${info.get('dayHigh', 'N/A')}")
            st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        with col2:
            st.metric("Open", f"${info.get('open', 'N/A')}")
            st.metric("Day Low", f"${info.get('dayLow', 'N/A')}")
            st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

        # Revenue and Market Cap
        st.write("**Market Cap:**", f"${info.get('marketCap', 'N/A'):,}")
        st.write("**Total Revenue:**", f"${info.get('totalRevenue', 'N/A'):,}")
        st.write("**Revenue per Share:**", f"${info.get('revenuePerShare', 'N/A')}")

        # Historical Chart
        st.subheader("Historical Chart")
        period = st.selectbox("Select Period", ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
        hist = stock.history(period=period)

        # Plot Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
        fig.update_layout(title=f"{ticker.upper()} Stock Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

    ### ML Part
        # Add prediction section
        st.subheader("Price Prediction")
        # Add date input for prediction
        min_date = hist.index.min().date()
        max_date = hist.index.max().date() + timedelta(days=365)  # Allow prediction up to 1 year in future
        user_date = st.date_input(
            "Select date for prediction",
            value=hist.index.max().date() + timedelta(days=30),  # Default to 30 days in future
            min_value=min_date,
            max_value=max_date
        )

        future_days = 10
        
        # Get 2 years of historical data for model training
        training_data = stock.history(period='2y') # Prepare data for the LSTM model
        training_data = training_data[['Close']]  # Use 'Close' prices for prediction
        scaler = StandardScaler()  # Normalize the data
        training_data_scaled = scaler.fit_transform(training_data)

        # Create train-test split
        look_back = 365
        X, y = [], []

        for i in range(look_back, len(training_data_scaled)):
            X.append(training_data_scaled[i - look_back:i, 0])  # Last 60 days
            y.append(training_data_scaled[i, 0])  # The next day's price

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

        # Split data into training and testing sets
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build the LSTM model
        model = Sequential([
            Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            Bidirectional(LSTM(units=50, return_sequences=False)),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)  # Final output layer
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        # Predict price for the chosen date
        future_date_ordinal = datetime.toordinal(user_date)  # Convert user_date to ordinal
        last_sequence = training_data_scaled[-look_back:]  # Last known sequence
        future_predictions = []  # Store predictions
        prediction_dates = []  # Store dates for prediction

        for i in range(future_days):
            prediction = model.predict(np.expand_dims(last_sequence, axis=0))[0, 0]
            future_predictions.append(prediction)
            
            # Append predicted date
            prediction_dates.append(user_date + timedelta(days=i))
            
            # Shift and append prediction
            last_sequence = np.vstack([last_sequence[1:], [[prediction]]])

        # Inverse transform the predictions to get actual prices
        predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

        # Display predicted prices for each day
        for i, predicted_price in enumerate(predicted_prices):
            st.write(f"Predicted Price on {prediction_dates[i].strftime('%Y-%m-%d')}: **${predicted_price:.2f}**")

        st.write("*Note: Prediction model is trained on 2 years of historical data*")

        predicted_test_prices = model.predict(X_test)

        # Inverse transform the predictions and actual values to get the real stock prices
        predicted_test_prices = scaler.inverse_transform(predicted_test_prices.reshape(-1, 1)).flatten()
        actual_test_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calculate Mean Squared Error (MSE) or Root Mean Squared Error (RMSE)
        mse = mean_squared_error(actual_test_prices, predicted_test_prices)
        rmse = math.sqrt(mse)

        # Optionally, you can also add Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual_test_prices, predicted_test_prices)

        st.subheader("Model Performance Evaluation")
        st.write(f"📊 **Model Accuracy (MSE)**: The difference between predicted and actual prices: **{mse:.2f}**")
        st.write(f"📉 **Model Prediction Error (RMSE)**: The average prediction error: **{rmse:.2f}**")
        st.write(f"🔍 **Average Prediction Deviation (MAE)**: How far off the predictions are, on average: **{mae:.2f}**")

        # User selection for ML prediction visualization type
        st.subheader("Visualization of ML Prediction")
        visualization_type = st.selectbox(
            "Select Visualization Type",
            ["Line Plot", "Seaborn Heatmap", "Bar Plot"]
        )

        if visualization_type == "Line Plot":
            # Line Plot using Matplotlib
            plt.figure(figsize=(12, 6))
            plt.plot(prediction_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
            plt.title(f"Stock Price Prediction for {ticker.upper()}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

        elif visualization_type == "Seaborn Heatmap":
            # Heatmap using Seaborn
            # Convert predictions into a format suitable for heatmap visualization (e.g., a 2D array)
            prediction_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Price': predicted_prices
            })
            prediction_pivot = prediction_df.pivot("Date", "Predicted Price", "Predicted Price")
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(prediction_pivot, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title(f"Stock Price Prediction Heatmap for {ticker.upper()}")
            st.pyplot(plt)

        elif visualization_type == "Bar Plot":
            # Bar Plot using Matplotlib
            plt.figure(figsize=(12, 6))
            plt.bar(prediction_dates, predicted_prices, color='skyblue')
            plt.title(f"Stock Price Prediction for {ticker.upper()}")
            plt.xlabel("Date")
            plt.ylabel("Predicted Price")
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt)


    except Exception as e:
        st.error(f"Error fetching data for {ticker.upper()}. Please check the ticker.")
    