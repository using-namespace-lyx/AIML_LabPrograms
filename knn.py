from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def calculate_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn_predict(X_train, y_train, x_test, k):
    distances = []

    for i in range(len(X_train)):
        distance = calculate_distance(x_test, X_train[i])
        distances.append((distance, y_train[i]))

    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:k]

    class_votes = {}

    for neighbor in nearest_neighbors:
        neighbor_class = neighbor[1]

        if neighbor_class in class_votes:
            class_votes[neighbor_class] += 1
        else:
            class_votes[neighbor_class] = 1

    predicted_class = max(class_votes, key=class_votes.get)
    return predicted_class

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dataset description
print("\nDescription of the Data Set")
print("Name of the Data Set: Iris")
print("No. of Features:", X.shape[1])
print("No. of Instances:", X.shape[0])
print("No. of Numerical Data: 4")  # Assuming all features are numerical
print("No. of Categorical Data: 0")  # Assuming there are no categorical features
print("Name of the Target Variable: Iris Species")

# KNN Algorithm
k = 3
print("\nPredicting class of the target variable")
for i in range(len(X_test)):
    x_test = X_test[i]
    predicted_class = knn_predict(X_train, y_train, x_test, k)
    print(f"Predicted class for {x_test}: {predicted_class}, Actual class: {y_test[i]}")
