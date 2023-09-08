#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from datetime import timedelta
import boto3
import os
from io import StringIO
import math
import mlflow
import mlflow.sklearn
from datetime import datetime

from utils import  comms  


# Initialise session state for authentication
if 'is_authenticated' not in st.session_state:
    st.session_state['is_authenticated'] = False


# Function to authenticate user using AWS Cognito
def authenticate(email, password):
    client = boto3.client('cognito-idp', region_name='eu-west-2')
    try:
        resp = client.initiate_auth(
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': password,
            },
            ClientId='ncv5lum49vqnk7m505e8pfvce' 
        )
        return resp.get("AuthenticationResult").get("IdToken")
    except Exception as e:
        st.warning("Failed to authenticate")
        return None



# Set environment variables
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS S3 stock bucket
stock_bucket = 'raw-stock-price'
# AWS S3 comment-section bucket
comment_bucket = 'comment-section-st'

#Function to add space in Streamlit layout.
def space(n: int):
    for _ in range(n):
        st.text('')


# Load data from S3
def load_data_from_s3(stock_name):
    file_name = f'yhoofinance-daily-historical-data/{stock_name}_daily_data.csv'
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)   
    obj = s3.get_object(Bucket=stock_bucket, Key=file_name)
    df = pd.read_csv(obj['Body'])
    return df

# Data preprocessing 
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Add features to the data
def add_feature(df, feature, window):
    if feature == 'MA':
        close_col = df['adj_close']
        df['MA'] = close_col.rolling(window=window).mean()
        df['MA'].fillna((df['MA'].mean()), inplace=True)
    if feature == 'EMA':
        close_col = df['adj_close']
        df['EMA'] = close_col.ewm(span=window, adjust=False).mean()
    if feature == 'STO':
        high14 = df['high'].rolling(window).max()
        low14 = df['low'].rolling(window).min()
        df['%K'] = (df['close'] - low14) * 100 / (high14 - low14)
        df['%K'].fillna((df['%K'].mean()), inplace=True)
        df['%D'] = df['%K'].rolling(3).mean()
        df['%D'].fillna((df['%D'].mean()), inplace=True)
    return df

# Train model function
def train_model(df, future_days, test_size, ma_window, ema_window, sto_window, opacity, features, stock_name):
    try:
    
        df['Prediction'] = df['adj_close'].shift(-future_days)

        df_copy = df.copy()

        X_predict = np.array(df_copy.drop(['Prediction'], 1))[-future_days:]
        X_predict = np.array(df.drop(['Prediction'], 1))[-future_days:]
      

        X = np.array(df.drop(['Prediction'], axis=1))
        X = X[:-future_days]
        y = np.array(df['Prediction'])
        y = y[:-future_days]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = LinearRegression()


        # Adding MLflow for MLOps
        with mlflow.start_run():
            mlflow.log_param("future_days", future_days)
            mlflow.log_param("test_size", test_size)

            model.fit(X_train, y_train)

            # Log model
            mlflow.sklearn.log_model(model, "linear_regression")

            # Log metrics: RMSE, MSE and MAPE
            rmse = math.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            mse = mean_squared_error(y_test, model.predict(X_test))
            mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAPE", mape)

            # Log additional parameters
            mlflow.log_param("MA", ma_window)
            mlflow.log_param("EMA", ema_window)
            mlflow.log_param("STO", sto_window)
            mlflow.log_param("opacity", opacity)
            mlflow.log_param("Features", features)
            mlflow.log_param("Stock Name", stock_name)
            mlflow.set_tag("user_id", st.session_state.user_id)

        model.fit(X_train, y_train)

        return model, X_train, X_test, y_train, y_test, X_predict



    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mlflow.end_run()

# Evaluate the model
def evaluate_model(model, X_test, y_test, metric):
    predictions = model.predict(X_test)
    if metric == 'rmse':
        return math.sqrt(mean_squared_error(y_test, predictions))
    elif metric == 'mse':
        return mean_squared_error(y_test, predictions)
    elif metric == 'mape':
        return mean_absolute_percentage_error(y_test, predictions)
    else:
        return None

# Plot the results
def plot_results(df, linear_model_real_prediction, linear_model_predict_prediction, display_at, future_days, opacity):
    predicted_dates = [df.index[-1] + timedelta(days=x) for x in range(1, future_days+1)]
    fig, ax = plt.subplots(figsize=(40, 20))
    plt.rcParams['figure.facecolor'] = 'black'
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.plot(df.index[display_at:], linear_model_real_prediction[display_at:], label='Linear Prediction', color='magenta', alpha=opacity, linewidth=5.0)
    ax.plot(predicted_dates, linear_model_predict_prediction, label='Forecast', color='aqua', alpha=opacity, linewidth=5.0)
    ax.plot(df.index[display_at:], df['adj_close'][display_at:], label='Actual', color='lightgreen', linewidth=5.0)

    # Plot MA, EMA, and STO if they exist in the dataframe
    if 'MA' in df.columns:
        ax.plot(df.index[display_at:], df['MA'][display_at:], label='MA', color='white', alpha=opacity, linewidth=10.0)
    if 'EMA' in df.columns:
        ax.plot(df.index[display_at:], df['EMA'][display_at:], label='EMA', color='red', alpha=opacity, linewidth=10.0)
    if '%K' in df.columns:
        ax.plot(df.index[display_at:], df['%K'][display_at:], label='%K', color='yellow', alpha=opacity, linewidth=10.0)
    if '%D' in df.columns:
        ax.plot(df.index[display_at:], df['%D'][display_at:], label='%D', color='blue', alpha=opacity, linewidth=10.0)
    
    date_format = DateFormatter("%Y-%m-%d")
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    plt.legend(prop={'size': 35})  #
    plt.xticks(fontsize=30)  
    plt.yticks(fontsize=30)  
    plt.show()


# Function to run the model
def run_model(stock_name, ma_window=5, ema_window=5, sto_window=5, features=['MA','close', 'EMA', 'STO'], test_size=0.5, future_days=30, rmse=True, mse=True, mape=True, display_at=0, opacity=0.5):
    model, evaluations, df = None, None, None

    try:
        st.write("Running model...")  
        # Load data from S3
        df = load_data_from_s3(stock_name)
        #st.write(f"loaded data: {df}")
        

        st.header("Data")
        # Preprocess data
        df = preprocess_data(df)
        st.write(df)
        

        # Define feature windows
        feature_windows = {
            'MA': ma_window,
            'EMA': ema_window,
            'STO': sto_window
        }

        # Add features to the data
        for feature in features:
            if feature in feature_windows:
                df = add_feature(df, feature, feature_windows[feature])
            #st.write(df)

        model, X_train, X_test, y_train, y_test, X_predict = train_model(df, future_days, test_size, ma_window, ema_window, sto_window, opacity, features, stock_name)
        st.write(f"Trained model: {model}")  

        # Train model and evaluate
        evaluations = {}
        if rmse:
            evaluations['rmse'] = evaluate_model(model, X_test, y_test, 'rmse')
        if mse:
            evaluations['mse'] = evaluate_model(model, X_test, y_test, 'mse')
        if mape:
            evaluations['mape'] = evaluate_model(model, X_test, y_test, 'mape')


        # Generate prediction
        linear_model_real_prediction = model.predict(np.array(df.drop(['Prediction'], 1)))
        linear_model_predict_prediction = model.predict(X_predict)

        # Plot
        plot_results(df, linear_model_real_prediction, linear_model_predict_prediction, display_at, future_days, opacity)


    except Exception as e:
        st.error(f"An error occurred: {e}")

    return model, evaluations, df


# Streamlit application
def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Check for authentication
    if not st.session_state['is_authenticated']:
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            token = authenticate(email, password)
            if token:
                st.session_state['token'] = token
                st.session_state['is_authenticated'] = True  # Set the session state
                st.success("Logged in")
            else:
                st.warning("Failed to log in: Please try again or return to https://main.dsxr40yvbyhag.amplifyapp.com to sign up")
        return 
    

    # Generate a unique ID for the user session
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
      
    col1, col2, col3 = st.sidebar.columns([1,2,1])
    with col1:
        st.image("assets/futurstox-high-resolution-logo-white-on-transparent-background.png", width=250)

    # Title
    st.title('Stock Price Prediction with Linear Regression')

    st.sidebar.markdown('# Parameters')
    stock_name = st.sidebar.selectbox("Select a stock", ("AAPL", "GOOGL", "MSFT", "AMZN","TSLA","META", "NFLX", "NVDA"), help="Select the stock for which you want to predict the price.")
    future_days = st.sidebar.slider('Days to Forecast', 1, 50, 30, help="Select the number of days in the future for which you want to predict the stock price.")
    display_at = st.sidebar.slider('Display From Day', 0, 365, 0, help="Select the starting day from which you want to display the stock price.")
    metrics = st.sidebar.multiselect('Evaluation Metrics', options=['RMSE', 'MSE', 'MAPE'], default=['RMSE', 'MSE', 'MAPE'], help="Select the metrics you want to use to evaluate the model's performance.")

    rmse = 'RMSE' in metrics
    mse = 'MSE' in metrics
    mape = 'MAPE' in metrics


    # Advanced Settings
    explanations = st.checkbox('Show Explanations')
    advanced_settings = st.sidebar.checkbox('Advanced Settings')
    if advanced_settings:
        ma_window = st.sidebar.slider('Moving Avg. -- Window Size', 1, 100, 50, help="Select the window size for the moving average.")
        ema_window = st.sidebar.slider('Exponential Moving Avg. -- Window Size', 1, 100, 50, help="Select the window size for the exponential moving average.")
        sto_window = st.sidebar.slider('Stochastic Oscillator -- Window Size', 1, 100, 50, help="Select the window size for the stochastic oscillator.")
        test_size = st.sidebar.slider('Test Set Size', 0.1, 0.9, 0.2, help="Select the size of the test set.")
        opacity = st.sidebar.slider('Opacity', 0.1, 1.0, 0.5, help="Select the opacity value for the model.")
        features = st.sidebar.multiselect('Features', options=['MA', 'EMA', 'STO', 'adj_close'], default=['adj_close'], help="Select the features you want to include in the model.")
    else:
        ma_window = 50
        ema_window = 50
        sto_window = 50
        test_size = 0.2
        opacity = 0.5
        features = ['adj_close']

    if st.sidebar.button('Train Model'):
        file_name = load_data_from_s3(stock_name)
        st.markdown('## Training Model...')
        model, evaluations, df = run_model(
            stock_name=stock_name,
            ma_window=ma_window,
            ema_window=ema_window,
            sto_window=sto_window,
            features=features,
            test_size=test_size,
            future_days=future_days,
            rmse=rmse,
            mse=mse,
            mape=mape,
            display_at=display_at,
            opacity=opacity
        )


        if model is not None and evaluations is not None and df is not None:
            # Display evaluation metrics in multiple columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header("RMSE")
                st.write(evaluations['rmse'])

            with col2:
                st.header("MSE")
                st.write(evaluations['mse'])

            with col3:
                st.header("MAPE")
                st.write(evaluations['mape'])

            st.markdown('## Forecast Plot')
            st.pyplot()
        else:
            st.error('An error occurred during model training')
    
    # Compare with previous performance
    if st.sidebar.button('Compare with Previous Performance'):
        runs = mlflow.search_runs(filter_string=f"tags.user_id = '{st.session_state.user_id}'")
        if len(runs) < 2:
            st.error("There are not enough runs to compare.")
        else:
            current_run = runs.iloc[0]
            previous_run = runs.iloc[1]

            # Get all parameter and metric columns
            param_cols = [col for col in current_run.index if col.startswith("params.")]
            metric_cols = [col for col in current_run.index if col.startswith("metrics.")]

            current_params = pd.DataFrame(current_run[param_cols]).T
            current_params.columns = [col.replace("params.", "") for col in current_params.columns]
            current_metrics = pd.DataFrame(current_run[metric_cols]).T
            current_metrics.columns = [col.replace("metrics.", "") for col in current_metrics.columns]

            previous_params = pd.DataFrame(previous_run[param_cols]).T
            previous_params.columns = [col.replace("params.", "") for col in previous_params.columns]
            previous_metrics = pd.DataFrame(previous_run[metric_cols]).T
            previous_metrics.columns = [col.replace("metrics.", "") for col in previous_metrics.columns]

            # Display the parameters and metrics of the current and previous runs
            st.header("Current Run:")
            st.write("**Parameters**:")
            st.table(current_params)
            st.write("**Metrics**:")
            st.table(current_metrics)

            st.header("*Previous Run*:")
            st.write("**Parameters**:")
            st.table(previous_params)
            st.write("**Metrics**:")
            st.table(previous_metrics)


    # Display explanations on the main page
    if explanations:
        st.markdown('## Explanations')
        st.markdown('**Evaluation Metrics**: Measures used to assess how well the model\'s predictions match the actual values.')
        st.markdown('- **RMSE (Root Mean Squared Error)**: A measure of the differences between the values predicted by the model and the actual values. Smaller values are better, with 0 being a perfect match.')
        st.markdown('- **MSE (Mean Squared Error)**: Similar to RMSE, but without taking the square root. This means larger errors are more heavily penalised.')
        st.markdown('- **MAPE (Mean Absolute Percentage Error)**: The average of the absolute percentage differences between the predicted and actual values. It gives an idea of the error rate in terms of the actual values.')
        if advanced_settings:
            st.markdown('**Window Size**: This is the number of consecutive data points used to calculate the feature. For example, if the window size is 5, the feature for the current day will be calculated using the data from the current day and the 4 previous days:')
            st.markdown('- **Moving Average**: This is the average stock price over the specified window of days. It helps to smooth out price fluctuations and highlight the overall trend.')
            st.markdown('- **Exponential Moving Average Window Size**: Similar to the moving average, but it gives more weight to recent prices. This makes it react more quickly to price changes.')
            st.markdown('- **Stochastic Oscillator Window Size**: This is a momentum indicator that compares a particular closing price of a stock to a range of its prices over a certain period of time. The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result.')


     # Connect to the S3 bucket
    s3 = comms.connect()
    comment_bucket = 'comment-section-st'
    file_name = 'linear_regression_comment/comments.csv'
    comments = comms.collect(s3, comment_bucket, file_name)


    # Add comment feature
    with st.expander("💬 Open comments"):
        # Show comments
        st.write("**Comments:**")

        for index, entry in enumerate(comments.itertuples()):
            st.markdown(f"**{entry.name}** ({entry.date}):\n\n&nbsp;\n\n&emsp;{entry.comment}\n\n---")


            is_last = index == len(comments) - 1
            is_new = "just_posted" in st.session_state and is_last
            if is_new:
                st.success("☝️ Your comment was successfully posted.")

        # Insert comment
        st.write("**Add your own comment:**")
        form = st.form("comment")
        name = form.text_input("Name")
        comment = form.text_area("Comment")
        submit = form.form_submit_button("Add comment")

        if submit:
            date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            comment.insert(s3, comment_bucket, file_name, [name, comment, date])
            if "just_posted" not in st.session_state:
                st.session_state["just_posted"] = True
            st.experimental_rerun()


if __name__ == '__main__':
    main()
