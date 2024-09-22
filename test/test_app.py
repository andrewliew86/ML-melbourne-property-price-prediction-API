import pytest
import json
from app import app

# Create a pytest fixture for the Flask app test client
@pytest.fixture
def client():
    app.testing = True  # Enable testing mode
    with app.test_client() as client:
        yield client

# Check if everything works as predicted
def test_predict_success(client):
    # Create a sample input for the /predict route
    sample_data = {
            'building_type': 'Unit',
            'rooms': 3,
            'showers': 2,
            'cars': 1,
            'size_clean': 120
    }

    # Send POST request to the /predict route
    response = client.post('/predict',
                           data=json.dumps(sample_data),
                           content_type='application/json')

    # Check if the response status code is 200
    assert response.status_code == 200

    # Parse the response data
    response_data = json.loads(response.data)

    # Check if the predicted price is in the response
    assert 'predicted_price' in response_data

# Check what happens when data are missing
def test_predict_missing_data(client):
    # Send request with missing data to test validation (showers)
    sample_data = {
            'building_type': 'Unit',
            'rooms': 3,
            'cars': 1,
            'size_clean': 120
    }
    response = client.post('/predict',
                           data=json.dumps(sample_data),
                           content_type='application/json')

    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert 'error' in response_data

# Check what happens when datatype is wrong
def test_predict_wrong_datatype(client):
    # Here size_clean is a string instead of an integer
    sample_data = {
            'building_type': 'Unit',
            'rooms': 3,
            'showers': 2,
            'cars': 1,
            'size_clean': "Hello"
    }

    response = client.post('/predict',
                           data=json.dumps(sample_data),
                           content_type='application/json')

    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert 'error' in response_data
    assert 'Invalid data type' in response_data['error']