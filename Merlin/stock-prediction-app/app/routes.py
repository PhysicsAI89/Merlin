from flask import Blueprint, render_template, request
from model.predict import load_data_and_model, make_prediction
import pandas as pd

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock']
    data, model = load_data_and_model(f'data/{stock_symbol}.csv', 'model/stock_model.h5')
    predictions = make_prediction(data, model)
    prediction_result = predictions[-1]  # Show the last prediction as an example
    return render_template('result.html', prediction=prediction_result)
