from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('house_price_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    building_type = data.get('building_type')
    rooms = data.get('rooms')
    showers = data.get('showers')
    garage = data.get('cars')
    property_size = data.get('size_clean')

    # Check if all data are present
    if not all([building_type, rooms, showers, garage, property_size]):
        return jsonify({'error': 'Missing data'}), 400

    # Validate the input types
    if not isinstance(rooms, (int, float)):        
        return jsonify({'error': 'Invalid data type for rooms. It must be a number.'}), 400
    if not isinstance(showers, (int, float)):        
        return jsonify({'error': 'Invalid data type for showers. It must be a number.'}), 400
    if not isinstance(garage, (int, float)):        
        return jsonify({'error': 'Invalid data type for garage. It must be a number.'}), 400
    if not isinstance(property_size, (int, float)):        
        return jsonify({'error': 'Invalid data type for property_size. It must be a number.'}), 400

    # Prepare the input dataframe
    input_data = pd.DataFrame([[building_type, rooms, showers, garage, property_size]], columns=['building_type_clean', 'room_clean', 'shower_clean', 'car_clean', 'size_clean'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
