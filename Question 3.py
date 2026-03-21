from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42)

breast_cancer_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
breast_cancer_model.fit(features_train, labels_train)

predicted_breast_cancer = breast_cancer_model.predict(features_test)

test_accuracy = accuracy_score(labels_test, predicted_breast_cancer)
print(f'Test Accuracy: {test_accuracy}')

train_accuracy = accuracy_score(labels_train, breast_cancer_model.predict(features_train))
print(f'Training Accuracy: {train_accuracy}')

top_five_features = pd.Series(
    breast_cancer_model.feature_importances_,
    index=breast_cancer_data.feature_names
).sort_values(ascending=False)

print(top_five_features.head(5))

#When max_depth is unrestricted, the model overfits by memorizing the training data, achieving 100% training accuracy
# but only 95.6% test accuracy. Setting max_depth to 3 reduced overfitting, bringing training and test accuracy closer
# together at 98% and 96.5%

#Feature importance improves the interpretability of decision trees by revealing which features drive predictions most.
# In this model mean concave points alone accounted for 66.5% of the decision making, followed by worst perimeter and
# worst texture, making it easy to understand which tumor measurements are most critical for diagnosis.
