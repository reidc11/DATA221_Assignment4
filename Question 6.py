import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

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

#CNNs are preferred over fully connected networks for image data because they detect spatial patterns like edges and
#shapes by sliding filters across the image, rather than treating each pixel independently. This makes them far more
#efficient and effective for image tasks.

#The convolution layer is learning to detect low level visual patterns in the clothing images such as edges, textures
#and curves that help the model distinguish between the different categories.

