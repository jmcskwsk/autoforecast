import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def main():
    # Load the Excel file
    df = pd.read_excel('Data_Sources/Full_Fiscal22-24_Consolidated.xlsx')

    # Extract date columns
    date_columns = df.columns[1:]

    # Convert date columns to numerical values for regression
    date_indices = np.arange(len(date_columns)).reshape(-1, 1)

    # Initialize a list to store the forecasted values
    forecasts = []

    # Iterate over each row to train a linear regression model
    for index, row in df.iterrows():
        sales_data = row[1:].values.reshape(-1, 1)
        
        # Train the linear regression model
        model = LinearRegression()
        model.fit(date_indices, sales_data)
        
        # Predict the next month's sales
        next_month_index = np.array([[len(date_columns)]])
        forecast = model.predict(next_month_index)
        
        # Append the forecasted value to the list
        forecasts.append(forecast[0][0])

    # Add the forecasted values as a new column in the DataFrame
    df['Forecast'] = forecasts

    # Save the updated DataFrame to a new Excel file
    print("Saving forecast to forecast.xlsx")
    df.to_excel('forecast.xlsx', index=False)

if __name__ == "__main__":
    main()