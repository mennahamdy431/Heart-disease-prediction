from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import joblib

catmodel = joblib.load('rand_model.pkl')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route('/predictHeartDisease', methods=["POST"])
def fun():
    if request.method == "POST":
        gender = request.form['gender']
        chest_pain = request.form['chest_pain']
        fasting_blood_sugar = request.form['fasting_blood_sugar']
        resting_ecg = request.form['resting_ecg']
        exercise_angina = request.form['exercise_angina']
        st_slope = request.form['st-slope']
        num_vessels_fluro = request.form['num-vessels-fluro']
        thallium = request.form['thallium']
        
        data = np.array([[st_slope, thallium,
                                num_vessels_fluro, fasting_blood_sugar, chest_pain,
                                gender, exercise_angina, resting_ecg]])
        prediction = catmodel.predict(data)
        print(prediction)
        return render_template('results.html', prediction=prediction)
    else:
        print('not printed')
        return  render_template('predict.html')

if __name__=="__main__":
    app.run(debug=True) 