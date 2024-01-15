import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
from pyswarm import pso
from joblib import dump

# Load data from Excel file
data = pd.read_excel('P3_Training_Data.xlsx')
X = data.iloc[:, :-1].values  # Input features
y = data.iloc[:, -1].values  # Output target

# Set up K-Fold Cross Validation
n_splits = 5  # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True)

best_model = None
best_mse = np.inf

# Perform K-Fold Cross Validation
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Fold {fold+1}:')
    
    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('Depth of Crack for Testing', y_test)
    # Define the objective function for PSO optimization
    def objective_function(params):
        gamma, C = params
        model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse
    
    # Perform PSO optimization
    lb = [0.001, 0.001]  # Lower bounds for gamma and C
    ub = [100, 100]  # Upper bounds for gamma and C
    parameters, _ = pso(objective_function, lb, ub)
    gamma_opt, C_opt = parameters
    
    # Train the SVR model with the optimal hyperparameters
    model = SVR(kernel='rbf', gamma=gamma_opt, C=C_opt, epsilon=0)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate error percentage
    error_percentage = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
    
    # Calculate MSE and accuracy percentage
    mse = mean_squared_error(y_test, y_pred)
    accuracy_percentage = 100 - error_percentage
    
    print('Prediction Output:', y_pred)
    print('Error Percentage:', error_percentage)
    print('MSE:', mse)
    print('Accuracy Percentage:', accuracy_percentage)
    
    # Save the best model based on the lowest MSE
    if mse < best_mse:
        best_model = model
        best_mse = mse
        best_percentage = accuracy_percentage

# Print the parameters of the best model
print('Best Model Choosen: ', best_percentage)
print('Best Model Parameters:')
print('Gamma:', best_model.gamma)
print('C:', best_model.C)

# Save the best PSO-SVR model
joblib.dump(best_model, 'P3_best_model.joblib')

# Load the best model
loaded_model = joblib.load('P3_best_model.joblib')

