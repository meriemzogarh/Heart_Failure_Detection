from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
app = Flask(__name__)

# Load the trained Random Forest model
with open('your_random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chest_pain_type = float(request.form['chest_pain_type'])
        resting_bp = float(request.form['resting_bp'])
        cholesterol = float(request.form['cholesterol'])
        fasting_bs = float(request.form['fasting_bs'])
        resting_ecg = float(request.form['resting_ecg'])
        max_hr = float(request.form['max_hr'])
        exercise_angina = float(request.form['exercise_angina'])
        oldpeak = float(request.form['oldpeak'])
        st_slope = float(request.form['st_slope'])

        # Create a feature array
        input_data = np.array([age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                              resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]).reshape(1, -1)

        # Use the loaded rf_model to make predictions
        prediction = rf_model.predict(input_data)

        # You can customize the response based on your prediction
        if prediction[0] == 0:
            result = 'Congratulations! No Heart Disease was detected in your body'
        else:
            result = 'We are sorry to inform you, Heart Disease was detected in your body'

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
