from functools import reduce
from keras import Input, Model
from keras.layers import RandomZoom, RandomRotation, RandomFlip, Dense, GlobalAveragePooling2D, Dropout, \
    BatchNormalization, Activation, Conv2D, MaxPool2D, AveragePooling2D, concatenate
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.optimizers import Adam

target_size = 299


class InceptionV3Model:
    def __init__(self):
        self.model = None
        self.target_size = None
        self.augmentation_layers = None

    def init(self):
        self.target_size = target_size

        self.augmentation_layers = [
            RandomZoom(height_factor=(-0.05, 0.15), fill_mode="reflect"),
            RandomRotation(15 / 360, fill_mode="reflect"), RandomFlip(mode="horizontal"),
        ]

        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.target_size, self.target_size, 3))
        augmented_input = reduce(lambda x, y: y(x), [input_layer, *self.augmentation_layers])
        x = self.stem(augmented_input)
        x = self.inception_blocks(x)
        x = GlobalAveragePooling2D()
        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(2, activation="softmax")
        model = Model(inputs=input_layer, outputs=x, name="Inception-V3")
        return model

    def stem(self, x):
        x = self.conv_block(x, 32, (3, 3), strides=(2, 2))
        x = self.conv_block(x, 32, (3, 3))
        x = self.conv_block(x, 64, (3, 3))
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = self.conv_block(x, 80, (1, 1))
        x = self.conv_block(x, 192, (3, 3))
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        return x

    def inception_blocks(self, x):
        x = self.inception_A(x, 32)
        x = self.inception_A(x, 64)
        x = self.inception_A(x, 64)
        x = self.reduction_A(x)
        x = self.inception_B(x, 128)
        x = self.inception_B(x, 160)
        x = self.inception_B(x, 160)
        x = self.inception_B(x, 192)
        x = self.reduction_B(x)
        x = self.inception_C(x)
        x = self.inception_C(x)
        return x

    def conv_block(self, x, filters, kernel_size, strides=(1, 1), padding="same"):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation(activation="relu")(x)
        return x

    def inception_A(self, x, kernels):
        br1 = self.conv_block(x, 64, (1, 1))
        br1 = self.conv_block(br1, 96, (3, 3))
        br1 = self.conv_block(br1, 96, (3, 3))

        br2 = self.conv_block(x, 48, (1, 1))
        br2 = self.conv_block(br2, 64, (3, 3))

        br3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        br3 = self.conv_block(br3, kernels, (1, 1))

        br4 = self.conv_block(x, 64, (1, 1))

        output = concatenate([br1, br2, br3, br4], axis=3)
        return output

    def inception_B(self, x, kernels):
        br1 = self.conv_block(x, kernels, (1, 1))
        br1 = self.conv_block(br1, kernels, (7, 1))
        br1 = self.conv_block(br1, kernels, (1, 7))
        br1 = self.conv_block(br1, kernels, (7, 1))
        br1 = self.conv_block(br1, 192, (1, 7))

        br2 = self.conv_block(x, kernels, (1, 1))
        br2 = self.conv_block(br2, kernels, (1, 7))
        br2 = self.conv_block(br2, 192, (7, 1))

        br3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        br3 = self.conv_block(br3, 192, (1, 1))

        br4 = self.conv_block(x, 192, (1, 1))

        output = concatenate([br1, br2, br3, br4], axis=3)
        return output

    def inception_C(self, x):
        br1 = self.conv_block(x, 448, (1, 1))
        br1 = self.conv_block(br1, 384, (3, 3))
        br1_first = self.conv_block(br1, 384, (1, 3))
        br1_second = self.conv_block(br1, 384, (3, 1))
        br1 = concatenate([br1_first, br1_second], axis=3)

        br2 = self.conv_block(x, 384, (1, 1))
        br2_first = self.conv_block(br2, 384, (1, 3))
        br2_second = self.conv_block(br2, 384, (3, 1))
        br2 = concatenate([br2_first, br2_second], axis=3)

        br3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        br3 = self.conv_block(br3, 192, (1, 1))

        br4 = self.conv_block(x, 320, (1, 1))

        output = concatenate([br1, br2, br3, br4], axis=3)
        return output

    def reduction_A(self, x):
        br1 = self.conv_block(x, 64, (1, 1))
        br1 = self.conv_block(br1, 96, (3, 3))
        br1 = self.conv_block(br1, 96, (3, 3), strides=(2, 2))

        br2 = self.conv_block(x, 384, (3, 3), strides=(2, 2))
        br3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        output = concatenate([br1, br2, br3], axis=3)
        return output

    def reduction_B(self, x):
        br1 = self.conv_block(x, 192, (1, 1))
        br1 = self.conv_block(br1, 192, (1, 7))
        br1 = self.conv_block(br1, 192, (7, 1))
        br1 = self.conv_block(br1, 192, (3, 3), strides=(2, 2), padding="valid")

        br2 = self.conv_block(x, 192, (1, 1))
        br2 = self.conv_block(br2, 320, (3, 3), strides=(2, 2), padding="valid")
        br3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
        output = concatenate([br1, br2, br3], axis=3)
        return output


inception_model = InceptionV3Model().model

inception_model.summary()

inception_model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])

history = inception_model.fit(train_images_array, train_labels, batch_size=32, epochs=55, validation_split=0.1)
