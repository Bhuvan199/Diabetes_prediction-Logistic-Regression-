import pickle
from flask import Flask,render_template,jsonify,request,Response
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# import classifier and scaler pickle file
model = pickle.load(open('models/classifier.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

# route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata',methods = ["GET","POST"])
def predict_datapoint():
    result = ""
    
    if request.method == "POST":
        Pregnancies = float(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))
    

        new_data_scaled=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = model.predict(new_data_scaled)

        if predict[0] ==1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"\
        
        return render_template("single_prediction.html",result = result)
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001)
