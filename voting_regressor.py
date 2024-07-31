import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning
import joblib

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

# Save the trained model to a file
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# Load the model from a file if it exists or create a new model if it doesn't
def load_model(model_path):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"No existing model found at {model_path}. A new model will be created.")
        return None

# Train the model and forecast the sales for each row in the DataFrame
def train_and_forecast(df, date_indices, model, model_path, param_grid):
    forecasts = []
    
    # Iterate over each row in the DataFrame, excluding the first column (Style)
    for index, row in df.iterrows():
        sales_data = row[1:].values.reshape(-1, 1) # Exclude the first column (Style)
        n_samples = len(sales_data)
        n_splits = min(5, n_samples)  # Ensure n_splits is not greater than n_samples
        
        # Adjust param_grid to ensure n_neighbors is not greater than n_samples
        adjusted_param_grid = {key: [min(n_samples, val) if key == 'n_neighbors' else val for val in values]
                               for key, values in param_grid.items()}
        
        # Create a new model for each row to avoid overwriting the previous model
        grid_search = GridSearchCV(model, adjusted_param_grid, cv=n_splits, n_jobs=-1, scoring='neg_mean_squared_error')
        
        # Perform grid search for hyperparameter tuning
        grid_search.fit(date_indices, sales_data.ravel())
        best_model = grid_search.best_estimator_
        print(f"Best parameters found for row {index}: {grid_search.best_params_}")

        # Train the best model on the entire dataset
        best_model.fit(date_indices, sales_data.ravel())
        next_month_index = np.array([[len(date_indices)]])
        forecast = best_model.predict(next_month_index)
        forecasts.append(int(forecast[0]))

    # Save the best model to a file
    save_model(best_model, model_path)
    return forecasts

# Save the forecasted sales back to the DataFrame and write it to an Excel file
def save_forecast(df, forecasts, model_name, output_dir):
    df['Forecast'] = forecasts
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{model_name}_forecast_{timestamp}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Forecast saved to {output_file}")

if __name__ == "__main__":
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    # Ignore warnings for UndefinedMetricWarning
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

    file_path = 'Data_Sources/Full_Fiscal22-24_Consolidated.xlsx'
    #file_path = 'Data_Sources/Train/train23.xlsx'
    output_dir = 'Predictions'
    model_path = 'Models/voting_regressor_model.pkl'

    df = load_data(file_path)
    date_columns = extract_date_columns(df)
    date_indices = convert_dates_to_indices(date_columns)

    svr = SVR()
    knn = KNeighborsRegressor()
    voting_regressor = VotingRegressor([('svr', svr), ('knn', knn)])

    # Define the parameter grid for SVR and KNN
    # Note: The number of neighbors should not exceed the number of samples
    # in the training data, so we adjust the parameter grid accordingly.
    svr_param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': [0.001, 0.01, 0.1, 0.2],
        'svr__kernel': ['rbf', 'linear', 'poly']
    }

    # Adjust the number of neighbors for KNN based on the number of samples
    knn_param_grid = {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['ball_tree', 'kd_tree']
    }

    # Combine the parameter grids for SVR and KNN
    param_grid = {**svr_param_grid, **knn_param_grid}

    forecasts = train_and_forecast(df, date_indices, voting_regressor, model_path, param_grid)
    save_forecast(df, forecasts, 'VotingRegressor', output_dir)