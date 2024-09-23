import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Load in cleaned property dataset 
df = pd.read_csv("melbourne-realestate-clean.csv")

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

# Preprocess the categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['building_type_clean'])
    ], remainder='passthrough')  # Leave the other features unchanged

# Create a pipeline with the preprocessor and RandomForestRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Define the hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, 
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)

# Predict on the test set
y_pred = random_search.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
