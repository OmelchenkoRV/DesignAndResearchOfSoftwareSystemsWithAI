import numpy as np
import tensorflow as tf
from keras.src.layers import Dense
import matplotlib.pyplot as plt

x = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

y = np.array([0, 1, 1, 0, 1, 0, 0, 1]).T

model = tf.keras.Sequential()
model.add(Dense(10, input_dim=3, activation="relu", name="Hidden"))
model.add(Dense(1, activation="sigmoid", name="Output"))

optimizer = tf.optimizers.Adam(learning_rate=0.05)
loss = "binary_crossentropy"
metrics = [tf.keras.metrics.BinaryAccuracy(),
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.AUC()]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
training = model.fit(x, y, epochs=200)
loss, accuracy, precision, recall, auc = model.evaluate(x, y, verbose=0)

print("Loss:", loss * 100, "%")
print("Accuracy:", accuracy * 100, "%")
print("Precision:", precision * 100, "%")
print("Recall:", recall * 100, "%")
print("AUC:", auc * 100, "%")

plt.plot(training.history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

predictions = model.predict(x)
print(predictions)
