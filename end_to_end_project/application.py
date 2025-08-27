from flask import Flask,render_template,request,jsonify,url_for,redirect
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler as SS
import numpy as np
import seaborn as sns


#import models
ridge_model = pickle.load(open('models/model1.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))
application = Flask(__name__)
app = application

@app.route('/')
def index():     
    return render_template("home.html")


@app.route('/predictFWI',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # return "ok"
        # print(request.form.get("Temperature"))
        # return "bye"
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Region = float(request.form.get('Region'))
        Fire = float(request.form.get('Fire'))
        newData = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Region,
       Fire]])
        res = ridge_model.predict(newData)
        return render_template('home.html', result = res[0])
    else:
        return render_template('home.html')


if(__name__ == "__main__"):
    app.run(debug=True)
