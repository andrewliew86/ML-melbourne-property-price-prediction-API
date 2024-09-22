# Here we will be testing our app by sending POST requests to our deployed Flask app (Docker container)
# Each request sends input data (building type, number of bathrooms etc...) and returns the predicted price
# Before running script below. Make sure:
# 1. Open Docker Desktop
# 2. Open powershell and cd to the folder containing all your code (i.e. 'cd .\path_to_folder\')
# 3. Type 'docker build -t flask-house-prices-app .' to build your Docker image
# 4. Type 'docker run -p 5000:5000 flask-house-prices-app' to run your Docker container



import requests
import json

# URL of the Flask app
url = 'http://localhost:5000/predict'

# The input data for the prediction
data = {
    'building_type': 'Unit',
    'rooms': 3,
    'showers': 2,
    'cars': 1,
    'size_clean': 120
}

# Send the POST request
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Print the prediction result
    prediction = response.json()
    print(f"Predicted price: {prediction['predicted_price']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
