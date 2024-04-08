import keras

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras import layers

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)
NUM_CLASSES = 10
cifar10_classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

x_train2 = (x_train / 255) - 0.5
x_test2 = (x_test / 255) - 0.5
y_train2 = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test2 = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential([
    layers.Conv2D(filters=96, kernel_size=(11, 11), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),

    layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),

    layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
y_pred_test = model.predict(x_test2)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)

cols = 5
rows = 2
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test[random_index, :])
        pred_label = cifar10_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        true_label = cifar10_classes[y_test[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
            pred_label, pred_proba, true_label
        ))
plt.show()
