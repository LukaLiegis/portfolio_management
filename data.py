import yfinance as yf
import pytz

def get_data(ticker):
    df = yf.Ticker(ticker).history(period="1y", auto_adjust=True).reset_index()
    df = df.rename(columns = {
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    df["datetime"] = df["datetime"].dt.tz_convert(pytz.utc)
    # Calculate the percent change os the close data
    df["returns"] = df["close"].pct_change()
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime", drop = True)
    df = df.dropna() 
    print("The size of ", ticker, "is ", df.shape)
    return df