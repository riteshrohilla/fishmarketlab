from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('fish_weight_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    species = int(request.form['species'])
    length = float(request.form['length'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])
    
    features = np.array([[species, length, length2, length3, height, width]])
    prediction = model.predict(features)
    
    return f'The predicted weight of the fish is: {prediction[0]:.2f} grams'

if __name__ == '__main__':
    app.run(debug=True)
`