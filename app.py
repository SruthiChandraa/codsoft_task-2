from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model and LabelEncoder
model = joblib.load('rf_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create an input array
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict the class
        prediction = model.predict(input_features)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f'Iris species predicted: {predicted_class}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
