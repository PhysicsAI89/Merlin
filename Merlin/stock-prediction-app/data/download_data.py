import yfinance as yf

def download_data(stock_symbol, start_date, end_date, file_path):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data.to_csv(file_path)

# Example usage
if __name__ == "__main__":
    download_data('AAPL', '2020-01-01', '2023-07-25', 'data/AAPL.csv')
