import os
import zipfile
import tensorflow as tf
import pandas as pd
from keras import Model
from keras.src.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer
zip_file_path = r'C:\Users\Desktop\Downloads\archive.zip'
directory_to_check = 'data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
    # print(zip_file.namelist())
    filelist = []
    labels = set()
    for dirname, _, filenames in os.walk('images'):

        for filename in filenames:
            filelist.append(os.path.join(dirname, filename))

for image in filelist:
    parts = image.split('\\')
    # Extract the second part (index 1) and add it to the set
    labels.add(parts[1])

unique_values_list = list(labels)
unique_values_list.sort()
print(len(filelist))
print(len(unique_values_list))

df = pd.DataFrame(filelist, columns=['source'])

df['label'] = df['source'].str.extract(r'\\([^\\]+)\\')

label_counts = df['label'].value_counts()

target_size = (224, 224)
shuffle = False
batch_size = 256
seed = 8
train, test = train_test_split(df, test_size=0.15)
val, test = train_test_split(test, test_size=0.75)

img_generator = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip='true')

x_train = img_generator.flow_from_dataframe(dataframe=train, x_col='source', y_col='label', target_size=target_size,
                                            batch_size=batch_size, shuffle=shuffle, seed=seed)
x_val = img_generator.flow_from_dataframe(dataframe=val, x_col='source', y_col='label', target_size=target_size,
                                          batch_size=batch_size, shuffle=shuffle, seed=seed)
x_test = img_generator.flow_from_dataframe(dataframe=test, x_col='source', y_col='label', target_size=target_size,
                                           batch_size=batch_size, shuffle=shuffle, seed=seed)


class ConcatenateLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ConcatenateLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)


def Inseption_net(x, filters):
    branch1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    branch3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    # Use the custom ConcatenateLayer to concatenate the branches
    return ConcatenateLayer(axis=3)([branch1x1, branch3x3, branch5x5, branch_pool])


input_layer = Input(shape=(224, 224, 3))

x = Conv2D(32, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Conv2D(32, (1, 1), strides=(2, 2), padding='same', activation='relu')(x)
x = Conv2D(96, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Inseption_net(x, [64, 96, 128, 16, 48, 64])
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Inseption_net(x, [192, 96, 228, 16, 48, 64])
x = Inseption_net(x, [160, 112, 224, 24, 64, 64])
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4)(x)

output_layer = Dense(len(unique_values_list), activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train,
    steps_per_epoch=x_train.samples // batch_size,
    validation_data=x_val,
    validation_steps=x_val.samples // batch_size,
    epochs=10
)
