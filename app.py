from flask import Flask, request, render_template, flash, redirect, url_for
import pickle
import pandas as pd
from datetime import datetime
import csv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret'

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))

users = {'user@example.com': 'password123'}

# Initialize the CSV file with headers if it doesn't exist
try:
    with open('predictions_log.csv', mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Timestamp', 'Pregnancies', 'Glucose', 'Blood Pressure', 
            'Skin Thickness', 'Insulin', 'BMI', 
            'Diabetes Pedigree Function', 'Age', 'Prediction'
        ])
except FileExistsError:
    pass

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in users and users[email] == password:
            flash('Login successful!', 'success')
            return redirect(url_for('home')) 
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    prediction = None  
    if request.method == 'POST':
        try:
            # Collect form data
            pregs = request.form.get('pregs', type=int)
            gluc = request.form.get('gluc', type=int)
            bp = request.form.get('bp', type=int)
            skin = request.form.get('skin', type=int)
            insulin = request.form.get('insulin', type=float)
            bmi = request.form.get('bmi', type=float)
            func = request.form.get('func', type=float)
            age = request.form.get('age', type=int)

            # Check if all fields are filled
            if None in (pregs, gluc, bp, skin, insulin, bmi, func, age):
                flash('All fields are required.', 'danger')
            else:
                # Prepare input for prediction
                input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
                prediction = model.predict(scaler.transform(input_features))[0]  

                # Log prediction to CSV file
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = [timestamp, pregs, gluc, bp, skin, insulin, bmi, func, age, prediction]
                
                with open('predictions_log.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(log_entry)

                # Flash prediction result
                result_msg = "Diabetic" if prediction == 1 else "Non-Diabetic"
                flash(f'Prediction made: {result_msg}', 'success')

        except ValueError:
            flash('Please provide valid input values', 'danger')
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
        
    return render_template('index.html', prediction=prediction)

@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    bmi_result = None
    if request.method == 'POST':
        try:
            weight = request.form.get('weight', type=float)
            height = request.form.get('height', type=float)

            if weight <= 0 or height <= 0:
                flash('Weight and height must be positive values.', 'danger')
            else:
                bmi_result = weight / (height ** 2)
                flash(f'Your BMI is {bmi_result:.2f}', 'success')
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
    
    return render_template('bmi.html', bmi_result=bmi_result)

@app.route('/data')
def data_analysis():
    df = pd.read_csv('diabetes.csv')

    data_table = df.to_html(classes="table table-striped", index=False)
    top_20_df = df.head(20)

    data_table = top_20_df.to_html(classes="table table-striped", index=False)

    return render_template('data.html', data_table=data_table)

if __name__ == '__main__':
    app.run(debug=True)