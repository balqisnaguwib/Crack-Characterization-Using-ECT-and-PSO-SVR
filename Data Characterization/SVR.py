import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read Excel file
data = pd.read_excel('P1_Training_Data.xlsx', header=None)

# Extract features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVR model
svr = SVR()
svr.fit(X_train, y_train)

# Predict on the test set
y_pred = svr.predict(X_test)

# Calculate error percentage
error_percentage = mean_absolute_error(y_test, y_pred) / y_test.mean() * 100

# Calculate accuracy percentage
accuracy_percentage = 100 - error_percentage

# Print error and accuracy percentages
print("Error Percentage: {:.2f}%".format(error_percentage))
print("Accuracy Percentage: {:.2f}%".format(accuracy_percentage))

# Additional evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: {:.2f}".format(mse))
print("R-squared Score: {:.2f}".format(r2))
