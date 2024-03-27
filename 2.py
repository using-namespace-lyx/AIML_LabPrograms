import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def calculate_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def preprocess(x):
    # Tokenize or preprocess the features (not specified, assuming no preprocessing for simplicity)
    return x

def fit_naive_bayes(X_train, y_train):
    class_probabilities = {}
    feature_probabilities = {}

    classes = np.unique(y_train)
    total_training_examples = len(X_train)

    for c in classes:
        X_c = X_train[y_train == c]
        class_probabilities[c] = len(X_c) / total_training_examples

        feature_probabilities[c] = {}
        total_features_in_class = len(X_c)

        for feature_index in range(X_train.shape[1]):
            unique_values, counts = np.unique(X_c[:, feature_index], return_counts=True)
            feature_probabilities[c][feature_index] = dict(zip(unique_values, counts / total_features_in_class))

    return class_probabilities, feature_probabilities

def predict_naive_bayes(X_test, class_probabilities, feature_probabilities):
    predictions = []
    probabilities = []

    for x_test in X_test:
        scores = {}

        for c in class_probabilities:
            score = np.log(class_probabilities[c])

            test_features = preprocess(x_test)

            for feature_index, feature_value in enumerate(test_features):
                if feature_index not in feature_probabilities[c]:
                    # Laplace smoothing
                    log_prob = np.log((1 + 1) / (total_features_in_class + len(feature_probabilities[c])))
                else:
                    # Check if the feature_value exists in the probabilities dictionary
                    if feature_value in feature_probabilities[c][feature_index]:
                        log_prob = np.log(feature_probabilities[c][feature_index][feature_value])
                    else:
                        # Laplace smoothing for unseen values
                        log_prob = np.log(1 / (total_features_in_class + len(feature_probabilities[c][feature_index])))

                # Update the class score for this test example
                score += log_prob

            scores[c] = score

        predicted_class = max(scores, key=scores.get)
        predictions.append(predicted_class)
        probabilities.append(scores)

    return predictions, probabilities

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Naive Bayes model
class_probabilities, feature_probabilities = fit_naive_bayes(X_train, y_train)

# Make predictions
predictions, probabilities = predict_naive_bayes(X_test, class_probabilities, feature_probabilities)

# Print predictions, actual classes, and probabilities
for i in range(len(X_test)):
    print(f"Predicted class for {X_test[i]}: {predictions[i]}, Actual class: {y_test[i]}, Probabilities: {probabilities[i]}")

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
