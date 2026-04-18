import pandas as pd

def create_features(df):

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek

    # Lag features (VERY IMPORTANT for forecasting)
    df["lag_1"] = df["energy"].shift(1)
    df["lag_2"] = df["energy"].shift(2)
    df["lag_24"] = df["energy"].shift(24)

    # Rolling mean (trend capture)
    df["rolling_mean_3"] = df["energy"].rolling(window=3).mean()

    # Drop missing rows created by shifting
    df = df.dropna()

    return df