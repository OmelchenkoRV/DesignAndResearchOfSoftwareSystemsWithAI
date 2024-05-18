import keras
import numpy as np
import tensorflow as tf
from keras import preprocessing, layers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras import applications
import matplotlib.pyplot as plt
from keras.src.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input


def image_generator(datagen, directory, target_size, batch_size, class_mode):
    gen = datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )
    while True:
        x_batch, y_batch = next(gen)
        yield x_batch, y_batch


def data_augmenter():
    data_augmentation = keras.Sequential([
        layers.Rescaling(1. / 255),
        layers.Resizing(299, 299),
        layers.RandomFlip('horizontal'),
        layers.RandomFlip('vertical'),
        layers.RandomRotation(factor=0.4, fill_mode="wrap"),
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="wrap"),
        layers.RandomBrightness(factor=0.2),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.RandomContrast(factor=0.2),
        layers.RandomCrop(height=224, width=224),
    ])
    return data_augmentation


data_augmentation = data_augmenter()

datagen_1 = ImageDataGenerator()
datagen_2 = ImageDataGenerator()
preprocess_input = keras.applications.inception_v3.preprocess_input

base_model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False
inputs = keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = keras.layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

train_it = datagen_1.flow_from_directory(r'C:\Users\Rostyslav\Downloads\archive\images\train',
                                         target_size=(299, 299),
                                         batch_size=32,

                                         shuffle=True,
                                         color_mode='rgb',
                                         class_mode='categorical')

valid_it = datagen_1.flow_from_directory(r'C:\Users\Rostyslav\Downloads\archive\images\valid',
                                         target_size=(299, 299),
                                         batch_size=32,

                                         shuffle=True,
                                         color_mode='rgb',
                                         class_mode='categorical')

# history1 = model.fit(train_it,
#                      validation_data=valid_it,
#                      # steps_per_epoch=480,  # 481
#                      # validation_steps=125,  # 125
#                      epochs=2)

model.evaluate(valid_it)

base_model.trainable = True
# for layer in base_model.layers:
#     if "BatchNormalization" in layer.__class__.__name__:
#         layer.trainable = True

# freeze = np.round((len(model.layers) - len(model.layers) * 0.3), 0).astype('int')
# for layer in model.layers[:freeze]:
#     layer.trainable = False
# for layer in model.layers[freeze:]:
#     layer.trainable = True
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

train_dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(datagen_2, r'C:\Users\Rostyslav\Downloads\archive\images\train', (299, 299), 32,
                            'categorical'),
    output_signature=(
        tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.int32)  # Assuming 5 classes
    )
)

valid_dataset = tf.data.Dataset.from_generator(
    lambda: image_generator(datagen_2, r'C:\Users\Rostyslav\Downloads\archive\images\valid', (299, 299), 32,
                            'categorical'),
    output_signature=(
        tf.TensorSpec(shape=(None, 299, 299, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.int32)  # Assuming 5 classes
    )
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

history = model.fit(train_it, validation_data=valid_it, epochs=6)
model.evaluate(valid_it)
model.save('Inception_v3_TL_.keras')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
