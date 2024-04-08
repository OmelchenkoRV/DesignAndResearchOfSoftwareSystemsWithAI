import keras
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras import Sequential, layers, datasets
from keras.src.layers import Dense
from keras.utils import to_categorical
mnist = datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# plt.imshow(x_train[0])
# plt.show()




model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1)

model.evaluate(x_test, y_test, verbose=2)
