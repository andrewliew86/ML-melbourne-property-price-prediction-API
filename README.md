# 🏠 House price prediction API

This POC project is a Flask-based API that predicts property prices in Melbourne. It leverages a machine learning model from scikit-learn that is trained from recent data scraped from a real-estate website. Categorical features (like building type) are preprocessed through a pipeline. Code was based on tutorial here: https://engineering.rappi.com/serve-your-first-model-with-scikit-learn-flask-docker-df95efbbd35e. 

The API is Dockerized, allowing easy deployment in any environment. Simply make POST requests with house feature data, and the model returns the predicted price.

## Features
- ""Property data**: Data (building type, price, land size, car spots etc...) extracted from a real-estate website
- **Machine Learning Model**: Uses a Random Forest regressor trained on house data
- **Categorical Preprocessing**: Categorical columns (e.g., `building_type`) are handled via scikit-learn’s `OneHotEncoder` pipeline
- **RESTful API**: Accepts JSON data for predictions and returns predicted house prices in real-time
- **Dockerized**: Can be deployed in any Docker-supported environment

## How It Works

1. **Input**: The API expects a JSON input with house features like building type, area, rooms, and garage availability. Example:
   ```json
   {
      "building_type": "Unit",
      "rooms": 3,
      "showers": 2,
      "cars": 1,
      "size_clean": 120
   }
   ```

2. **Prediction**: The input is processed via a scikit-learn pipeline. The model predicts the price based on the provided features.

3. **Output**: The API responds with a JSON object containing the predicted price.

   Example response:
   ```json
   {
     "predicted_price": 350000
   }
   ```

**Figure 1: Basic overview of the API**

<img src="https://github.com/andrewliew86/Melbourne-property-price-prediction-model/blob/main/images/schematic.png" width=70% height=70%>

## Project Structure
- **`propert-scraping-script.py`**: Python code for obtaining data
- **`exploratory-data-analysis-script.py`**: Scripts for exploratory analysis of data (building type, price, land size, car spots etc...) extracted from a real-estate website prior to model training
- **`model-training-script.py`**: Scripts for model training
- **`app.py`**: The main Flask app that handles the API requests and returns predictions.
- **`model.pkl`**: The serialized Random Forest regressor model.
- **`Dockerfile`**: Docker configuration to containerize the app.
- **`test_app.py`**: Unit tests using pytest to ensure the app handles valid and invalid data correctly.

## Getting Started

### Prerequisites
- **Docker**: Ensure you have Docker installed on your machine.
- **Python 3.9**: Required for local development (if not using Docker).

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/andrewliew86/Melbourne-property-price-prediction-model.git
   cd Melbourne-property-price-prediction-model
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t flask-house-prices-app .
   ```

3. **Run the Docker Container**:
   ```bash
   docker run -p 5000:5000 flask-house-prices-app
   ```

### API Usage

Once the container is running, you can send a POST request to the `/predict` endpoint.

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"building_type": "Unit", "rooms": 3, "showers": 2, "cars": 1, "size_clean": 120}'
```

**Example Response**:
```json
{
  "predicted_price": 350000
}
```

### Running Tests
Run the tests using pytest to ensure everything works as expected:
```bash
pytest test_app.py
```

## Example results from exploratory data analysis

**Figure 2: Folium map showing locations of properties**

<img src="https://github.com/andrewliew86/Melbourne-property-price-prediction-model/blob/main/images/propery_folium_map.PNG" width=70% height=70%>

**Figure 3: Counts of property types, frequently used words to descrive properties and distribution of prices and built year**

<img src="https://github.com/andrewliew86/Melbourne-property-price-prediction-model/blob/main/images/fig3.png" width=70% height=70%>

## Future Improvements
- Implement additional machine learning models for better accuracy
- Hyperparameter tuning of models
- Deploy the API to a cloud platform like AWS or Heroku for scaling
