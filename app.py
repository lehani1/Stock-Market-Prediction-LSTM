from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Keras

from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'SBIN.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')
from datetime import datetime, timedelta,date

class NSE():
    def __init__(self, timeout=10):
        self.base_url = 'https://www.nseindia.com'
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.43",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "accept-language": "en-US,en;q=0.9"
        }
        self.timeout = timeout
        self.cookies = []

    def __getCookies(self, renew=False):
        if len(self.cookies) > 0 and renew == False:
            return self.cookies

        r = requests.get(self.base_url, timeout=self.timeout, headers=self.headers)
        self.cookies = dict(r.cookies)
        return self.__getCookies()

    def getHistoricalData(self, symbol, series, from_date, to_date):
        try:
            url = "/api/historical/cm/equity?symbol={0}&series=[%22{1}%22]&from={2}&to={3}&csv=true".format(symbol.replace('&', '%26'), series, from_date.strftime('%d-%m-%Y'), to_date.strftime('%d-%m-%Y'))
            r = requests.get(self.base_url + url, headers=self.headers, timeout=self.timeout, cookies=self.__getCookies())
            if r.status_code != 200:
                r = requests.get(self.base_url + url, headers=self.headers, timeout=self.timeout, cookies=self.__getCookies(True))

            df = pd.read_csv(BytesIO(r.content), sep=',', thousands=',')
            df = df.rename(columns={ 'Date ': 'date','series ': 'series', 'OPEN ': 'open', 'HIGH ': 'high', 'LOW ': 'low', 'PREV. CLOSE ': 'prev_close', 'ltp ': 'ltp', 'close ': 'close', '52W H ': 'hi_52_wk', '52W L ': 'lo_52_wk', 'VOLUME ': 'trdqty', 'VALUE ': 'trdval', 'No of trades ': 'trades'})
            df.date = pd.to_datetime(df.date)
            df.set_index("date", inplace = True)
            df.sort_index(inplace = True)
            return df
        except:
            return None



def get_date_years_back(years):
    current_date = datetime.now()
    delta = timedelta(days=365 * years)
    target_date = current_date - delta
    return target_date


def create_chart(df):
    plt.figure(figsize=(16,6))
    plt.title("Close Price History")
    plt.plot(df["close"])
    plt.xlabel("date", fontsize=18)
    plt.ylabel("Close Price (Lakhs INR)", fontsize = 18)
    plt.savefig("static/"+ "closingchart.png")

def get_data(stock):
    nse = NSE()
    today = datetime.now()
    start = get_date_years_back(1)
    df = nse.getHistoricalData(stock, 'EQ', date(start.year,start.month,start.day), date(today.year,today.month,today.day))
    return df


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def prediction(model,data):
    dataset = data[-10:].filter(["close"]).values
    scaler = MinMaxScaler(feature_range=(0,1))
    s_data = scaler.fit_transform(dataset)
    preds = model.predict(s_data)
    real_vals = scaler.inverse_transform(preds)
    plt.figure(figsize=(16,6))
    plt.title("Model Predictions")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(data['close'])
    plt.plot(real_vals)
    plt.legend(['Real', 'Predicted'], loc='lower right')
    plt.savefig("static/" + "predicted.png")
    return real_vals

if __name__ == '__main__':
    data = get_data(stock="SBIN")
    create_chart(data)
    preds = prediction(model, data)
    print(preds)
    app.run(debug=True)

