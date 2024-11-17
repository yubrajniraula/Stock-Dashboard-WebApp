import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

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

        # Get 2 years of historical data for model training
        training_data = stock.history(period='2y')
        
        # Prepare data for ML model using the 2-year data
        training_data['Date'] = training_data.index
        training_data['Date'] = training_data['Date'].map(datetime.toordinal)  # Convert dates to ordinal numbers
        X = np.array(training_data['Date']).reshape(-1, 1)
        y = np.array(training_data['Close'])

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict price for the chosen date
        chosen_date_ordinal = datetime.toordinal(user_date)
        predicted_price = model.predict([[chosen_date_ordinal]])

        # Display the prediction
        st.write(f"Predicted Price on {user_date.strftime('%Y-%m-%d')}: **${predicted_price[0]:.2f}**")
        st.write("*Note: Prediction model is trained on 2 years of historical data*")

    except Exception as e:
        st.error(f"Error fetching data for {ticker.upper()}. Please check the ticker.")
