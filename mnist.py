import keras
import numpy as np
from keras import initializers
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import UpSampling2D
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def show_train_image(image_index):
    print(y_train[image_index])
    plt.imshow(x_train[image_index], cmap='Greys')
    plt.show()

def show_test_image(image_index):
    print(y_test[image_index])
    plt.imshow(x_test[image_index], cmap='Greys')
    plt.show()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

input_shape = x_train[0].shape
num_classes = 10
batch_size = 128
epochs = 2

np.random.seed(42)
my_init = initializers.glorot_uniform(seed=42)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

visible = Input(shape=input_shape)

encode = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(visible)
encode = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(encode)
encode = MaxPooling2D(pool_size=(2, 2))(encode)
encode = Dropout(0.25)(encode)
encode = Flatten()(encode)
encode = Dense(128, activation='relu')(encode)
encode = Dropout(0.25)(encode)

output = Dense(num_classes, name='class', activation='softmax')(encode)

decode = Dense(12544, activation='relu')(encode)
decode = Reshape((14, 14, 64))(decode)
decode = Dropout(0.25)(decode)
decode = UpSampling2D(size=(2, 2))(decode)
decode = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(decode)
decode = Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same', name='decoded', kernel_initializer=my_init)(decode)

model = Model(inputs=visible, outputs=[output, decode])
model.summary()

model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
              optimizer=keras.optimizers.Adadelta(),
              metrics={'class' : 'accuracy', 'decoded' : 'accuracy'},
              loss_weights = [1, 1])

model.fit(x_train, [y_train, x_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, [y_test, x_test]))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
