import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load the data from the Excel file into the DataFrame
def load_data(file_path):
    return pd.read_excel(file_path)

# Extract the date columns from the DataFrame
def extract_date_columns(df):
    return df.columns[1:]

# Convert the date columns to indices for training the model
def convert_dates_to_indices(date_columns):
    # Convert date columns to datetime and format them
    formatted_dates = pd.to_datetime(date_columns).strftime('%y-%b')
    return np.arange(len(formatted_dates)).reshape(-1, 1)

# Train the model and forecast the sales for each row in the DataFrame
def train_and_forecast(df, date_indices, model):
    forecasts = []
    # Iterate over each row in the DataFrame, excluding the first column (Style)
    for index, row in df.iterrows():
        sales_data = row[1:].values.reshape(-1, 1)
        model.fit(date_indices, sales_data.ravel())  # Use ravel() to flatten the array
        next_month_index = np.array([[len(date_indices)]])
        forecast = model.predict(next_month_index)
        forecasts.append(int(forecast[0]))  # Convert to integer directly
    return forecasts

# Save the forecasted sales back to the DataFrame and write it to an Excel file
def save_forecast(df, forecasts, model_name, output_dir):
    df['Forecast'] = forecasts
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/{model_name}-forecast_{timestamp}.xlsx' # Save the forecast with a timestamp
    print(f"Saving forecast to {filename}")
    df.to_excel(filename, index=False)

# Forecast the sales using the specified model and save the results
def forecast_with_model(model, model_name):
    file_path = 'Data_Sources/Full_Fiscal22-24_Consolidated.xlsx'
    output_dir = 'Predictions'
    
    df = load_data(file_path)
    date_columns = extract_date_columns(df)
    date_indices = convert_dates_to_indices(date_columns)
    forecasts = train_and_forecast(df, date_indices, model)
    save_forecast(df, forecasts, model_name, output_dir)

# Linear Regression
def linear_regression():
    model = LinearRegression()
    forecast_with_model(model, 'LinearRegression')

# Ridge Regression
def ridge():
    model = Ridge()
    forecast_with_model(model, 'Ridge')

# Lasso Regression
def lasso():
    model = Lasso()
    forecast_with_model(model, 'Lasso')

# Elastic Net Regression
def elastic_net():
    model = ElasticNet()
    forecast_with_model(model, 'ElasticNet')

# Decision Tree Regressor
def decision_tree():
    model = DecisionTreeRegressor()
    forecast_with_model(model, 'DecisionTreeRegressor')

# Random Forest Regressor
def random_forest():
    model = RandomForestRegressor()
    forecast_with_model(model, 'RandomForestRegressor')

# Gradient Boosting Regressor
def gradient_boosting():
    model = GradientBoostingRegressor()
    forecast_with_model(model, 'GradientBoostingRegressor')

# Support Vector Regressor
def svr():
    model = SVR()
    forecast_with_model(model, 'SVR')

# K-Nearest Neighbors Regressor
def k_neighbors():
    model = KNeighborsRegressor()
    forecast_with_model(model, 'KNeighborsRegressor')

# Run the forecast functions for each model
if __name__ == "__main__":
    linear_regression()
    ridge()
    lasso()
    elastic_net()
    decision_tree()
    random_forest()
    gradient_boosting()
    svr()
    k_neighbors()