import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras.optimizers import Adam
from tensorflow.keras import Sequential, Input, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y_fn = lambda x: np.sin((x * x) ** 0.5) + np.cos(3 * x / 2)
z_fn = lambda x, y: x * np.sin(x + y)

x_val = np.linspace(0, 10, 250)
y_val = y_fn(x_val)
X, Y = np.meshgrid(x_val, y_val)
Z = z_fn(X, Y)

scaler = StandardScaler()
x_input = np.column_stack((scaler.fit_transform(X.reshape(-1, 1)), scaler.fit_transform(Y.reshape(-1, 1))))
y_input = scaler.fit_transform(Z.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.2, random_state=42)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, y_input.reshape(-1, 250), cmap='plasma')
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
plt.show()


# Feed forward backprop

def create_model_ffb(hidden_neurons):
    model = Sequential([
        layers.Input(shape=[2], name='input_layer'),
        layers.Dense(hidden_neurons, activation='relu', name='hidden_layer'),
        layers.Dense(1, name='output_layer')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


def train_model(model):
    return model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0)


def plot_history(history):
    plt.plot(history.history['loss'], label='Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train'])
    plt.show()


def plot_result(model):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, np.reshape(model.predict(x_input), (-1, 250)), cmap='plasma')
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title('Model result')
    plt.show()


# Creating the models with different numbers of hidden neurons


number_hidden_neurons = [10, 20]

[y for x in number_hidden_neurons if (y := create_model_ffb(x).summary())]

[y for x in number_hidden_neurons if (y := plot_history(train_model(create_model_ffb(x))))]

[y for x in number_hidden_neurons if
 (y := print(f'Test loss for feed forward with {x} neurons'), create_model_ffb(x).evaluate(X_test, y_test))]

[y for x in number_hidden_neurons if (y := plot_result(create_model_ffb(x)))]

input_layer = Input(shape=[2, ], name='input_layer')
hidden_layer = layers.Dense(20, activation='relu', name='hidden_layer')(input_layer)
concateneted_layer = layers.concatenate([hidden_layer, input_layer])
output_layer = layers.Dense(1, name='output_layer')(concateneted_layer)
model_sfb_1 = Model(input_layer, output_layer, name='model_sfb_1')
model_sfb_1.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model_sfb_1.summary()

input_layer = Input(shape=[2, ], name='input_layer')
hidden_layer_1 = layers.Dense(10, activation='relu', name='hidden_layer_1')(input_layer)
concateneted_layer_1 = layers.concatenate([hidden_layer_1, input_layer])
hidden_layer_2 = layers.Dense(10, activation='relu', name='hidden_layer_2')(input_layer)
concateneted_layer_2 = layers.concatenate([hidden_layer_2, hidden_layer_1, input_layer])
output_layer = layers.Dense(1, name='output_layer')(concateneted_layer_2)
model_sfb_2 = Model(input_layer, output_layer, name='model_sfb_2')
model_sfb_2.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model_sfb_2.summary()

history_sfb_1 = train_model(model_sfb_1)
history_sfb_2 = train_model(model_sfb_2)

score_sfb_1 = model_sfb_1.evaluate(X_test, y_test)
score_sfb_2 = model_sfb_2.evaluate(X_test, y_test)

print('Test loss for Cascade Forward - 20 neurons:', score_sfb_1)
print('Test loss for Cascade Forward -  2 layers with 10 neurons:', score_sfb_2)

plot_history(history_sfb_1)
plot_history(history_sfb_2)

plot_result(model_sfb_1)
plot_result(model_sfb_2)

model_eb_1 = Sequential([
    layers.Input(shape=[2, 1], name='input_layer'),
    layers.SimpleRNN(15, activation='relu', name='hidden_layer'),
    layers.Dense(1, name='output_layer')
])

model_eb_1.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model_eb_1.summary()

model_eb_2 = Sequential([
    layers.Input(shape=[2, 1], name='input_layer'),
    layers.SimpleRNN(5, activation='relu', return_sequences=True, name='hidden_layer_1'),
    layers.SimpleRNN(5, activation='relu', return_sequences=True, name='hidden_layer_2'),
    layers.SimpleRNN(5, activation='relu', name='hidden_layer_3'),
    layers.Dense(1, name='output_layer')
])

model_eb_2.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model_eb_2.summary()

history_eb_1 = train_model(model_eb_1)
history_eb_2 = train_model(model_eb_2)

score_eb_1 = model_eb_1.evaluate(X_test, y_test)
score_eb_2 = model_eb_2.evaluate(X_test, y_test)

print('Test loss for Elman - 15 neurons:', score_eb_1)
print('Test loss for Elman -  3 layers with 5 neurons:', score_eb_2)

plot_history(history_eb_1)
plot_history(history_eb_2)

plot_result(model_sfb_1)
plot_result(model_eb_2)
