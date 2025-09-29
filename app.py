import numpy as np
from flask import Flask, request, render_template
import joblib

# Create the app
app = Flask(__name__)

# Load your model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# This is the main page
@app.route('/')
def home():
    return render_template('index.html')

# This function runs when you click the "Predict" button
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the numbers from the form
        cgpa = float(request.form['cgpa'])
        iq = float(request.form['iq'])

        # Prepare the features for the model
        input_features = np.array([[cgpa, iq]])
        scaled_features = scaler.transform(input_features)

        # Make a prediction
        prediction = model.predict(scaled_features)

        # Create the result text
        if prediction[0] == 1:
            result = "Placed 🎉"
        else:
            result = "Not Placed 😔"

    except Exception as e:
        result = f"Error: Please enter valid numbers."

    # Send the result back to the webpage
    return render_template('index.html', prediction_text=f'Prediction: The student will likely be {result}')

if __name__ == "__main__":
    app.run(debug=True)