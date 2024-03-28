import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_to_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def k_means(X, k, max_iterations=100):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iterations):
        labels = assign_to_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

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

# K-means Clustering Algorithm
k = 3  # Number of clusters
labels, centroids = k_means(X_train, k)

# Predicted classes of test samples
test_labels = assign_to_clusters(X_test, centroids)

# Display predicted and actual classes of test samples
for i in range(len(X_test)):
    predicted_class = test_labels[i]
    actual_class = y_test[i]
    print(f"Sample {i + 1}: Predicted Class - {predicted_class}, Actual Class - {actual_class}")

# Plot clusters with circle markers and legend
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X_train[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', alpha=0.7)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', s=80, c='black', label='Test Samples')

# Annotate predicted classes of test samples on the plot
for i in range(len(X_test)):
    plt.annotate(f'Predicted: {test_labels[i]}', (X_test[i, 0], X_test[i, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.title('K-means Clustering with Test Samples')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
# Calculate and print accuracy
accuracy = np.sum(test_labels == y_test) / len(y_test) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
