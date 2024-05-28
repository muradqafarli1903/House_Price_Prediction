from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('house_rent_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    bhk = int(request.form['BHK'])
    size = int(request.form['Size'])
    bathroom = int(request.form['Bathroom'])
    city = request.form['City']
    furnishing_status = request.form['Furnishing Status']
    tenant_preferred = request.form['Tenant Preferred']
    area_type=request.form['Area Type']
    point_of_contact=request.form['Point of Contact']

    # Create input DataFrame for the model
    input_data = pd.DataFrame({
        'BHK': [bhk],
        'Size': [size],
        'Bathroom': [bathroom],
        'City': [city],
        'Furnishing Status': [furnishing_status],
        'Tenant Preferred': [tenant_preferred],
        'Area Type': [area_type],
        'Point of Contact': [point_of_contact]
    })

    # Predict the rent
    prediction = model.predict(input_data)
    
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimated House Rent: ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


