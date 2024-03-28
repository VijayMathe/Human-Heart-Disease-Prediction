from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickel
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == "POST":
        
        Age = float(request.form.get('age'))
        Sex = float(request.form.get('sex'))
        CP = float(request.form.get('cp'))
        Trestbps = float(request.form.get('trestbps'))
        Chol = float(request.form.get('chol'))
        FBS = float(request.form.get('fbs'))
        Restecg = float(request.form.get('restecg'))
        Thalach = float(request.form.get('thalach'))
        Exang = float(request.form.get('exang'))
        Oldpeak = float(request.form.get('oldpeak'))
        CA = float(request.form.get('ca'))
        Thal = float(request.form.get('thal'))

        new_data_scaled = standard_scaler.transform([[Age,Sex,CP,Trestbps,Chol,FBS,Restecg,Thalach,Exang,Oldpeak,CA,Thal]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result = result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
