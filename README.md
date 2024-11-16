# Stock Dashboard WebApp

## Overview
This is a simple stock dashboard web application that provides:
- Stock details such as company name, current price, opening price, day low/high, 52-week low/high, market cap, revenue details, and latest news.
- Balance sheet (yearly and quarterly).
- Chart visualization of stock performance over selected periods.

## Features
- Search for any stock by its ticker.
- View detailed stock information.
- Fetch the latest company news.
- Switch between yearly and quarterly balance sheets.
- Interactive stock price chart for various time periods.

## Requirements
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/stock-dashboard.git
2. Navigate to the project directory:
    ```bash
    cd stock-dashboard
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
Or Create a Virtual Environment (Recommended)\
To avoid dependency conflicts in the future, use a virtual environment for your project:
    
        ```bash
        python -m venv env
        source env/bin/activate   # On Linux/Mac
        env\Scripts\activate      # On Windows
        pip install -r requirements.txt

## Run the App
1. Launch the Streamlit app:
    ```bash
    streamlit run app.py
2. Open the browser and navigate to:
    ```arduino
    http://localhost:8501