import keras
import numpy as np
from keras import initializers
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import UpSampling2D
from keras.losses import categorical_crossentropy
from keras import regularizers
from keras import backend as K
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
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)[:, :, :, :]
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
epochs = 10

np.random.seed(42)
my_init = initializers.glorot_uniform(seed=42)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

visible = Input(shape=input_shape)

# encode = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(visible)
# encode = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(encode)
# encode = MaxPooling2D(pool_size=(2, 2))(encode)
# encode = Dropout(0.25)(encode)
# encode = Flatten()(encode)
# encode = Dense(128, activation='relu')(encode)
# encode = Dropout(0.25)(encode)

# output = Dense(num_classes, name='class', activation='softmax')(encode)

# decode = Dense(12544, activation='relu')(encode)
# decode = Reshape((14, 14, 64))(decode)
# decode = Dropout(0.25)(decode)
# decode = UpSampling2D(size=(2, 2))(decode)
# decode = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(decode)
# decode = Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same', name='reconstruction', kernel_initializer=my_init)(decode)



# add a Dense layer with a L1 activity regularizer
encode = Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(visible)
flattened = Flatten()(encode)
output = Dense(num_classes, name='class', activation='softmax')(flattened)

decode = Dense(784, activation='sigmoid')(encode)







def conditional_categorical_crossentropy(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    # this loss functions gives zero loss when there is no label
    # otherwise, categorical_crossentropy
    return K.switch(K.flatten(K.equal(K.sum(y_true, axis=-1), 0.)), K.zeros_like(loss), loss)

def train_regularized_model(n_samples_train):
    model = Model(inputs=visible, outputs=[output, decode])
    model.summary()

    n = n_samples_train

    # train model with n_samples_train labelled data, rest un-labelled
    y_train_pruned = y_train
    y_train_pruned[n:,:] = 0

    model.compile(loss=[conditional_categorical_crossentropy, 'binary_crossentropy'],
                  optimizer=keras.optimizers.Adadelta(),
                  metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                  loss_weights = [1, 1])

    score = model.evaluate(x_test, [y_test, x_test], verbose=0)

    model.fit(x_train[0:n*2,:,:,:],
              [y_train_pruned[0:n*2], x_train[0:n*2,:,:,:]],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
              # validation_data=(x_test, [y_test, x_test]))

    score = model.evaluate(x_test, [y_test, x_test], verbose=0)

    print('Test prediction loss:', score[0])
    print('Test prediction accuracy:', score[2])
    print('Test reconstruction loss:', score[1])
    print('Test reconstruction accuracy:', score[3])

def train_basic_model(n_samples_train):
    model = Model(inputs=visible, outputs=output)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train[:n_samples_train,:,:,:],
              y_train[:n_samples_train,:],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
              # validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


x_test = x_test[:10000,:,:,:]
y_test = y_test[:10000,:]

train_basic_model(1000)
train_regularized_model(1000)
