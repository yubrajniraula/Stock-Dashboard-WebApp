import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# App Configuration
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# Title and Search Bar
st.title("📈 Stock Dashboard")
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

        # Latest News
        st.subheader("Latest News")
        for news in stock.news[:3]:
            st.write(f"- [{news['title']}]({news['link']})")

        # Balance Sheet
        st.subheader("Balance Sheet")
        balance_type = st.radio("View Balance Sheet:", ["Yearly", "Quarterly"])
        if balance_type == "Yearly":
            st.dataframe(stock.balance_sheet)
        else:
            st.dataframe(stock.quarterly_balance_sheet)

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

        future_days = 30
        
        # Get 2 years of historical data for model training
        training_data = stock.history(period='2y') # Prepare data for the LSTM model
        training_data = training_data[['Close']]  # Use 'Close' prices for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize the data
        training_data_scaled = scaler.fit_transform(training_data)

        # Create train-test split
        look_back = 60  # Use the past 60 days to predict the next price
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
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
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

        #Plot the historical and predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(prediction_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
        plt.title(f"Stock Price Prediction for {ticker.upper()}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        # Show the plot in Streamlit
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error fetching data for {ticker.upper()}. Please check the ticker.")
    
    # Prepare data for ML model using the 2-year data
        # training_data['Date'] = training_data.index
        # training_data['Date'] = training_data['Date'].map(datetime.toordinal)  # Convert dates to ordinal numbers
        # X = np.array(training_data['Date']).reshape(-1, 1)
        # y = np.array(training_data['Close'])

        # Train a Linear Regression model
    #     model = LinearRegression()
    #     model.fit(X, y)

    #     # Predict price for the chosen date
    #     chosen_date_ordinal = datetime.toordinal(user_date)
    #     predicted_price = model.predict([[chosen_date_ordinal]])

    #     # Display the prediction
    #     st.write(f"Predicted Price on {user_date.strftime('%Y-%m-%d')}: **${predicted_price[0]:.2f}**")
    #     st.write("*Note: Prediction model is trained on 2 years of historical data*")