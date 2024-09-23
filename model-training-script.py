# Here we will be training a decision regressor model with our property data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


import joblib
import math

# Load the dataset
df = pd.read_csv('melbourne-realestate-clean.csv')

# Given the lack of data, we will be taking a practical approach of only modelling using avaialble data and not try to impute values
# Lets remove commercial buildings (we are only interest in residential) and remove year built and size columns as there are loads of missing values in these
df = df[~df['building_type_clean'].isin(['Commercial'])]
df = df[['price_clean',	
         'building_type_clean',
         'room_clean',	
         'shower_clean',	
         'car_clean',	
         'size_clean']]

# Now remove rows with any NaNs
df.dropna(subset=['price_clean', 'building_type_clean', 'room_clean', 'shower_clean', 'car_clean', 'size_clean'], inplace=True)

# Define features and target
X = df[['building_type_clean', 'room_clean', 'shower_clean', 'car_clean', 'size_clean']] 
y = df['price_clean']  # Target variable

# Here, we will loop through and check MSE over several regression models
# Preprocess the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['building_type_clean'])
    ], remainder='passthrough')  # Keep other features unchanged

# List of regression models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Gradient boosted Regression': GradientBoostingRegressor()
}

# 3-fold cross-validation with preprocessing pipeline
for name, model in models.items():
    # Define the model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Perform 3-fold cross-validation and calculate the mean MSE for each model
    scores = cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    mean_error = math.sqrt(mean_mse)
    print(f"{name} Mean error from cross-fold val: {mean_error:.2f}")


## Model training
# Turns out random forest is the best so we are going with that
# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'house_price_model.joblib')

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
