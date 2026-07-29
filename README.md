# Natural Gas Price Forecasting

I built this as part of a quant research exercise where the task was: given monthly natural gas price snapshots, figure out a way to estimate the price on *any* date — including dates up to a year past the last available data point. The use case is pricing storage contracts, where you sometimes need an indicative price further out than your data actually covers.

## The problem

The data I had was monthly, end-of-month snapshots from October 2020 to September 2024. Natural gas prices trend upward over time but also swing seasonally — higher in winter (more heating demand), lower in summer. So a straight line through the data isn't enough; you need to capture both things at once.

## How I approached it

1. Fit a simple linear trend across the whole dataset to capture the general direction prices are moving.
2. Look at what's left over after removing that trend (the residuals) and group them by calendar month. This shows how far above or below trend each month typically sits — e.g. January tends to run higher, July tends to run lower.
3. To predict a price for any date: take the trend value for that point in time, then add the typical seasonal adjustment for that month.

It's a pretty simple model on purpose — no machine learning, just regression and averaging — but it captures the two things that actually matter here (long-term trend + seasonality) without overfitting to noise.

## Running it

```python
from datetime import datetime
from price_model import load_data, fit_trend, compute_monthly_offsets, predict_price

df = load_data('Nat_Gas.csv')
reference_date = df['Dates'][0]
trend_params = fit_trend(df)
monthly_offsets = compute_monthly_offsets(df, trend_params)

price = predict_price(datetime(2025, 3, 31), trend_params, monthly_offsets, reference_date)
print(price)
```

## Data

The CSV just needs two columns:
- `Dates` — end-of-month date
- `Prices` — market price on that date

## What I'd improve with more time

- Smooth the seasonality into a continuous curve (e.g. a sine fit) instead of fixed monthly averages, so it can interpolate between month-ends too.
- Add a plot comparing actual vs. predicted prices, since eyeballing the fit is honestly the fastest way to sanity-check a model like this.
- Handle multiple years of seasonal data more robustly if the dataset ever gets larger (right now it just averages residuals per month across however many years are available).

## Stack

Python, pandas, numpy
