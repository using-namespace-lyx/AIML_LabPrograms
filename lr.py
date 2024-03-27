from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def initialize_parameters(features):
    weights = np.random.rand(features)
    bias = np.random.rand()
    return weights, bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y, a):
    m = len(y)
    cost = - (1 / m) * (y.T @ np.log(a) + (1 - y).T @ np.log(1 - a))
    return cost

def compute_gradients(X, y, a):
    m = len(y)
    dw = (1 / m) * (X.T @ (a - y))
    db = (1 / m) * np.sum(a - y)
    return dw, db

def update_parameters(weights, bias, dw, db, learning_rate):
    weights -= learning_rate * dw
    bias -= learning_rate * db
    return weights, bias

def logistic_regression_train(X_train, y_train, learning_rate=0.01, iterations=1000):
    m, features = X_train.shape
    weights, bias = initialize_parameters(features)

    for iteration in range(1, iterations + 1):
        z = X_train @ weights + bias
        a = sigmoid(z)

        cost = compute_cost(y_train, a)
        dw, db = compute_gradients(X_train, y_train, a)

        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{iterations}, Cost: {cost}")

    print("Training completed.")
    return weights, bias

def logistic_regression_predict(X_test, weights, bias):
    z = X_test @ weights + bias
    predictions = sigmoid(z)
    return predictions

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # Binary classification, 1 if Iris-Virginica, 0 otherwise

# Display dataset description
print("Description of the Data Set:")
print(iris.DESCR)
print("Name of the Data Set: Iris")
print("No. of Features:", X.shape[1])
print("No. of Instances:", X.shape[0])
print("No. of Numerical Data:", X.shape[1])
print("No. of Categorical Data: 0")
print("Name of the Target Variable: Iris-Virginica (1) or Not (0)")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
learned_parameters = logistic_regression_train(X_train, y_train)

# Example usage for predictions
predictions = logistic_regression_predict(X_test, *learned_parameters)

# Display predictions and predicted class
for i in range(len(X_test)):
    prediction = predictions[i]
    predicted_class = 1 if prediction > 0.5 else 0
    print(f"Predicted probability for {X_test[i]}: {prediction:.4f}, Predicted class: {predicted_class}, Actual class: {y_test[i]}")