import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

(features_train, labels_train), (features_test, labels_test) = fashion_mnist.load_data()

features_train = features_train / 255.0
features_test = features_test / 255.0

features_train = features_train.reshape(-1, 28, 28, 1)
features_test = features_test.reshape(-1, 28, 28, 1)

fashion_model = Sequential([Input(shape=(28, 28, 1)), Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
                             Flatten(), Dense(64, activation='relu'), Dense(10, activation='softmax')])

fashion_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

fashion_model.fit(features_train, labels_train, epochs=15, validation_data=(features_test, labels_test))

test_loss, test_accuracy = fashion_model.evaluate(features_test, labels_test)
print(f"Test accuracy: {test_accuracy:.4f}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

predictions = fashion_model.predict(features_test)
predicted_labels = predictions.argmax(axis=1)

ConfusionMatrixDisplay.from_predictions(labels_test, predicted_labels,
    display_labels=class_names)
plt.show()

misclassified = np.where(predicted_labels != labels_test)[0]

print(f"Number of misclassified images: {len(misclassified)}")
print(f"First 3 misclassified indices: {misclassified[:3]}")

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

for i, idx in enumerate(misclassified[:3]):
    axes[i].imshow(features_test[idx].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {class_names[labels_test[idx]]}\nPred: {class_names[predicted_labels[idx]]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

#A pattern observed in the misclassifications is that the model struggles most with visually similar clothing categories
#such as t-shirts, shirts, pullovers, and Coats since they are all  similar shapes and with similar features at 28x28 pixel
#resolution.

#One realistic method to improve CNN performance would be to add more convolutional layers, allowing the model to learn
#more complex and detailed features from the images.