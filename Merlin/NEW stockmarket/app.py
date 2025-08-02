@app.route('/')
def index():
    return render_template('index.html')
    
from flask import request, redirect, url_for

@app.route('/import_stock_data', methods=['POST'])
def import_stock_data():
    # Call the stock data import function
    import_stock_data_function()  # Define this function in commands/stocks.py
    return redirect(url_for('index'))

@app.route('/analyze_stocks', methods=['POST'])
def analyze_stocks():
    # Call the stock analysis function
    analyze_stocks_function()  # Define this function in commands/analyze.py
    return redirect(url_for('index'))
