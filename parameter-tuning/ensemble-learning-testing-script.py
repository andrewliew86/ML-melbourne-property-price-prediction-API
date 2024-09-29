# Here I am testing whether ensembling gives better predictions
# Ensembling is a technique where the predictions of several base models are used as features by a final 'meta' model for predictions
# Ensembling can often lead to better predictions. See: https://medium.com/@brijesh_soni/stacking-to-improve-model-performance-a-comprehensive-guide-on-ensemble-learning-in-python-9ed53c93ce28

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load cleaned property dataset
df = pd.read_csv("melbourne-realestate-clean.csv")

# Remove commercial buildings and irrelevant columns
df = df[~df['building_type_clean'].isin(['Commercial'])]
df = df[['price_clean', 
         'building_type_clean', 
         'room_clean', 
         'shower_clean', 
         'car_clean', 
         'size_clean']]

# Remove rows with any NaNs
df.dropna(subset=['price_clean', 'building_type_clean', 'room_clean', 'shower_clean', 'car_clean', 'size_clean'], inplace=True)

# Define features and target
X = df[['building_type_clean', 'room_clean', 'shower_clean', 'car_clean', 'size_clean']] 
y = df['price_clean']  # Target variable

# Preprocess the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['building_type_clean'])
    ], remainder='passthrough')  # Keep numerical features unchanged

# Define base models for stacking
base_models = [
    ('rfr', RandomForestRegressor(random_state=42)),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('lr', LinearRegression())
]

# Define meta-model for stacking
meta_model = RandomForestRegressor(random_state=42)

# Create the Stacking Regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Create a pipeline with preprocessor and stacking regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', stacking_regressor)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Root mean Squared Error: {np.sqrt(mse)}")
