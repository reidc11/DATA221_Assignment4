from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


breast_cancer_data = load_breast_cancer()

feature_matrix = breast_cancer_data.data
target_vector = breast_cancer_data.target

features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, target_vector, test_size=0.2, random_state=42)

breast_cancer_decisiontree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
breast_cancer_decisiontree.fit(features_train, labels_train)

predicted_breast_cancer_decisiontree = breast_cancer_decisiontree.predict(features_test)



scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

breast_cancer_neuralnetwork = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',max_iter=500, random_state=42)
breast_cancer_neuralnetwork.fit(features_train_scaled, labels_train)
predicted_breast_cancer_neuralnetwork = breast_cancer_neuralnetwork.predict(features_test_scaled)

ConfusionMatrixDisplay.from_predictions(labels_test, predicted_breast_cancer_decisiontree,
    display_labels=['Benign', 'Malignant'])

plt.title('Confusion Matrix — Decision Tree')
plt.show()

ConfusionMatrixDisplay.from_predictions(labels_test, predicted_breast_cancer_neuralnetwork,
    display_labels=['Benign', 'Malignant'])

plt.title('Confusion Matrix — Neural Network')
plt.show()


#I would prefer the neural network for this task. Both models only made one mistake in identifying a malignant tumour as
#benign, but the neural network performed better on the benign cases overall.

#An advantage to decision trees is that it is easy to understand whats going on when the model runs. A downside is that
#decision trees are prone to overfitting because they will keep growing and memorizing unless you manually restrict them
#with something like max_depth.

#An advantage of neural networks is that they have a higher accuracy. Downside is that it is hard to explain and understand
#what the model is actually doing. Everything is in the "black box"