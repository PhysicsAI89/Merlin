import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Example usage
if __name__ == "__main__":
    data = preprocess_data('data/AAPL.csv')
    print(data.head())
