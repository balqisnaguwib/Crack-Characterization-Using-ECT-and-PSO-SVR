import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the data from Excel file
data = pd.read_excel('P1_Training_Data.xlsx', header=None)

# Extract the input features (all columns except the last one)
X = data.iloc[:, :-1].values

# Extract the output target (last column)
y = data.iloc[:, -1].values

# Split the data into training and test sets (70% for training, 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define and train the PSO-SVR model
model = SVR(kernel='rbf')

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the error percentage
error_percentage = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100

# Calculate the accuracy percentage
accuracy_percentage = 100 - error_percentage

# Print the error percentage and accuracy percentage
print('Error Percentage: {:.2f}%'.format(error_percentage))
print('Accuracy Percentage: {:.2f}%'.format(accuracy_percentage))

# Additional evaluation metrics
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: {:.2f}".format(mse))

