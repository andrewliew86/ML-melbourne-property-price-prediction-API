import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

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

# Plot feature importance plot
# Get the best model from the RandomizedSearchCV
best_model = random_search.best_estimator_

# Extract the OneHotEncoder step and the regressor
preprocessor = best_model.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = preprocessor.get_feature_names_out(['building_type_clean'])
regressor = best_model.named_steps['model']
importances = regressor.feature_importances_

# Now get the numeric column names (the passthrough columns)
numeric_feature_names = X.columns.drop('building_type_clean')

# Combine OneHotEncoder feature names and numeric column names
feature_names = list(ohe_feature_names) + list(numeric_feature_names)

# Create a DataFrame to store the feature importance and feature names
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot the top N important features
top_n = len(feature_names)  # You can adjust this number
plt.figure(figsize=(20, 6))
plt.barh(feature_importance_df['feature'][:top_n], feature_importance_df['importance'][:top_n], color='b')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from RandomForest model')
plt.gca().invert_yaxis() 
plt.savefig('feature-importance.png')
plt.show()
## Room size appears to be important but given the small number of features. We should interpret with caution.



# Generate the learning curve to determine if model is overfitting
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, 
                                                        cv=5, scoring='neg_mean_squared_error', 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), 
                                                        n_jobs=-1)

# Convert negative MSE to RMSE (Root Mean Squared Error)
train_rmse = np.sqrt(-train_scores) 
test_rmse = np.sqrt(-test_scores)

# Compute mean and standard deviation of RMSE for training and validation sets
train_rmse_mean = np.mean(train_rmse, axis=1)
train_rmse_std = np.std(train_rmse, axis=1)
test_rmse_mean = np.mean(test_rmse, axis=1)
test_rmse_std = np.std(test_rmse, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_rmse_mean, label="Training error", color="r")
plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std, train_rmse_mean + train_rmse_std, color="r", alpha=0.1)
plt.plot(train_sizes, test_rmse_mean, label="Cross-validation error", color="g")
plt.fill_between(train_sizes, test_rmse_mean - test_rmse_std, test_rmse_mean + test_rmse_std, color="g", alpha=0.1)

# Plot labels and title
plt.title('Learning Curve for RandomForest Regressor')
plt.xlabel('Training Set Size')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend(loc="best")
plt.grid()

# Save and show the plot
plt.savefig('learning_curve.png')
plt.show()
# Learning curve suggests some overfitting given the 'large' distance between training and validation loss at the end of the curve
# Rigorous hyperparam tuning, changing to a simpler model and/or obtaining more data might help with this