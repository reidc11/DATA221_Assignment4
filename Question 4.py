from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

breast_cancer_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',max_iter=500, random_state=42)
breast_cancer_model.fit(features_train_scaled, labels_train)

train_accuracy = accuracy_score(labels_train, breast_cancer_model.predict(features_train_scaled))
test_accuracy  = accuracy_score(labels_test,  breast_cancer_model.predict(features_test_scaled))

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:     {test_accuracy:.4f}")

#The features in this dataset all have different scales. Neural networks learn by adjusting weights using gradient
# descent, and this process works poorly when features have very different scales, as large values dominate the updates.

#An epoch is one complete pass through the entire training dataset, during which the network updates its weights to
# reduce its error
