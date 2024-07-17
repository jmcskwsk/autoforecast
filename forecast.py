import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def load_data(file_path):
    return pd.read_excel(file_path)

def extract_date_columns(df):
    return df.columns[1:]

def convert_dates_to_indices(date_columns):
    return np.arange(len(date_columns)).reshape(-1, 1)

def train_and_forecast(df, date_indices, model):
    forecasts = []
    for index, row in df.iterrows():
        sales_data = row[1:].values.reshape(-1, 1)
        model.fit(date_indices, sales_data)
        next_month_index = np.array([[len(date_indices)]])
        forecast = model.predict(next_month_index)
        forecasts.append(int(forecast[0].item()))  # Convert to integer
    return forecasts

def save_forecast(df, forecasts, model_name, output_dir):
    df['Forecast'] = forecasts
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/{model_name}-forecast_{timestamp}.xlsx'
    print(f"Saving forecast to {filename}")
    df.to_excel(filename, index=False)

def forecast_with_model(model, model_name):
    file_path = 'Data_Sources/Full_Fiscal22-24_Consolidated.xlsx'
    output_dir = 'Predictions'
    
    df = load_data(file_path)
    date_columns = extract_date_columns(df)
    date_indices = convert_dates_to_indices(date_columns)
    forecasts = train_and_forecast(df, date_indices, model)
    save_forecast(df, forecasts, model_name, output_dir)

def linear_regression():
    model = LinearRegression()
    forecast_with_model(model, 'LinearRegression')

def ridge():
    model = Ridge()
    forecast_with_model(model, 'Ridge')

def lasso():
    model = Lasso()
    forecast_with_model(model, 'Lasso')

def elastic_net():
    model = ElasticNet()
    forecast_with_model(model, 'ElasticNet')

if __name__ == "__main__":
    linear_regression()
    ridge()
    lasso()
    elastic_net()