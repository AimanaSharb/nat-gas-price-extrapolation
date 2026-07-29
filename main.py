import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    # striping the columns
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    print(df.columns)

    # converting the dates and sorting them(even if the data is already sorted, for accurate results)
    df["Dates"] = pd.to_datetime(df['Dates'])
    df= df.sort_values(by='Dates')
    df = df.reset_index(drop=True)


    reference_date = df["Dates"][0]
    df['MonthIndex'] = df['Dates'].apply(lambda x: date_to_numeric(x, reference_date))

    print(df)


    return df



def date_to_numeric(date, reference_date):
    date_month = date.month
    date_year = date.year
    reference_date_year = reference_date.year
    reference_date_month = reference_date.month

    formula = ((date_year - reference_date_year)*12) + (date_month - reference_date_month )
    return formula



def fit_trend(df):
    x = df['MonthIndex']
    y = df['Prices']

    result = np.polyfit(x, y, 1)
    print(result)
    return result



def compute_monthly_offsets(df, trend_params):    
    df['Predicted_Price'] =  (df["MonthIndex"]* trend_params[0]) + trend_params[1]
    df['Residual'] = df['Prices'] - df['Predicted_Price']
    df['Month'] = df['Dates'].dt.month
    montly_means = df.groupby('Month')['Residual'].mean()
    offsets = montly_means.to_dict()
    return offsets


def predict_price(date, trend_params, monthly_offsets, reference_date):
    month_index = date_to_numeric(date, reference_date)
    trend_price = (month_index * trend_params[0]) + trend_params[1]
    seasonal_offset = monthly_offsets[date.month]
    predicted_price = trend_price + seasonal_offset

    return predicted_price






def main():
    df = load_data('Nat_Gas.csv')
    reference_date = df['Dates'][0]
    trend_params = fit_trend(df)
    monthly_offsets = compute_monthly_offsets(df, trend_params)




    test_date = datetime(2027, 5, 31)
    price = predict_price(test_date, trend_params, monthly_offsets, reference_date)
    print(f"Estimated price on {test_date.date()}: {price}")

if __name__ == '__main__':
    main()