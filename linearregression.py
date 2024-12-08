        ### ML Part
        # Add prediction section
        st.subheader("Price Prediction")

        # Prepare data for prediction
        two_years_data = stock.history(period='2y')
        two_years_data = two_years_data[['Close', 'Volume']]
        two_years_data['50_MA'] = two_years_data['Close'].rolling(window=50).mean()
        two_years_data['200_MA'] = two_years_data['Close'].rolling(window=200).mean()
        two_years_data.dropna(inplace=True)  # Remove rows with NaN values due to moving averages

        # Features and target
        X = two_years_data[['Volume', '50_MA', '200_MA']].values
        y = two_years_data['Close'].values

                # Add date input for prediction
        min_date = two_years_data.index.min().date()  # Earliest available data
        max_date = two_years_data.index.max().date() + timedelta(days=365)  # Allow prediction up to 1 year in the future
        default_date = two_years_data.index.max().date() + timedelta(days=30)  # Default to 30 days in the future
        user_date = st.date_input(
            "Select date for prediction",
            value=default_date,
            min_value=min_date,
            max_value=max_date
        )

        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict for the test set
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        mse = np.mean((y_test - y_pred) ** 2)
        st.write(f"Mean Squared Error on Test Data: **{mse:.2f}**")

        # Prediction for user-specified date
        if user_date:
            st.write(f"You have selected: {user_date.strftime('%Y-%m-%d')}")
            
            # Handle extrapolation
            if user_date > two_years_data.index.max().date():
                # Extrapolate features for the future date
                last_known_data = two_years_data.iloc[-1]
                future_close = last_known_data['Close']
                future_volume = last_known_data['Volume']
                future_50_ma = last_known_data['50_MA']
                future_200_ma = last_known_data['200_MA']
                prediction_features = np.array([future_close, future_volume, future_50_ma, future_200_ma]).reshape(1, -1)
            else:
                # Find the nearest index in existing data
                user_date_datetime = pd.to_datetime(user_date)
                prediction_date_index = two_years_data.index.get_loc(user_date_datetime, method='nearest')
                prediction_features = X[prediction_date_index].reshape(1, -1)

            # Predict the price
            predicted_price = model.predict(prediction_features)[0]
            st.write(f"Predicted Price on {user_date.strftime('%Y-%m-%d')}: **${predicted_price:.2f}**")

        # # Visualize actual vs predicted prices
        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(two_years_data.index[train_size:], y_test, label='Actual Price', color='blue')
        # ax.plot(two_years_data.index[train_size:], y_pred, label='Predicted Price', color='red', linestyle='--')
        # ax.set_title(f"{ticker.upper()} Actual vs Predicted Prices")
        # ax.set_xlabel("Date")
        # ax.set_ylabel("Price")
        # ax.legend()
        # ax.grid(True)
        # st.pyplot(fig)