from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42)

breast_cancer_model = DecisionTreeClassifier(criterion='entropy')
breast_cancer_model.fit(features_train, labels_train)

predicted_breast_cancer = breast_cancer_model.predict(features_test)

test_accuracy = accuracy_score(labels_test, predicted_breast_cancer)
print(f'Test Accuracy: {test_accuracy}')

train_accuracy = accuracy_score(labels_train, breast_cancer_model.predict(features_train))
print(f'Training Accuracy: {train_accuracy}')

#The model achieved 100% training accuracy but 95.6% test accuracy which suggests mild overfitting. The decision
# tree memorized the training data rather than learning patterns that generalize well to new data.


#In decision trees, entropy measures how mixed the classes are at a given node. If a node contains only one
# class, entropy is 0. If the classes are evenly split 50/50, entropy is at its maximum.