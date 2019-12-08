import keras
import numpy as np
from keras import initializers
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras import backend as K

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def show_image(image):
    plt.imshow(image.reshape(28, 28), cmap='Greys')
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
batch_size = 1
epochs = 400

np.random.seed(42)
my_init = initializers.glorot_uniform(seed=42)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def create_model(regularized):
    visible = Input(shape=input_shape)

    encode = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=my_init)(visible)
    encode = MaxPooling2D(pool_size=(2, 2))(encode)
    # encode = Dropout(0.25)(encode)
    encode = Flatten()(encode)
    encode = Dense(64, activation='relu')(encode)
    # encode = Dropout(0.25)(encode)

    output = Dense(num_classes, name='class', activation='softmax')(encode)

    decode = Dense(1568, activation='relu')(encode)
    decode = Reshape((14, 14, 8))(decode)
    # decode = Dropout(0.25)(decode)
    decode = UpSampling2D(size=(2, 2))(decode)
    decode = Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same', name='reconstruction', kernel_initializer=my_init)(decode)

    if regularized:
        return Model(inputs=visible, outputs=[output, decode])
    else:
        return Model(inputs=visible, outputs=output)

def conditional_categorical_crossentropy(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred)
    # this loss functions gives zero loss when there is no label
    # otherwise, categorical_crossentropy
    return K.switch(K.flatten(K.equal(K.sum(y_true, axis=-1), 0.)), K.zeros_like(loss), loss)

def zero_loss(y_true, y_pred):
    return y_pred * 0

def train_regularized_model(n_samples_train):
    model = create_model(True)
    model.summary()

    n = n_samples_train

    # train model with n_samples_train labelled data, rest un-labelled
    y_train_pruned = y_train
    y_train_pruned[n:,:] = 0

    model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                  optimizer=Adam(clipnorm = 1.),
                  metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                  loss_weights={'class' : 0, 'reconstruction' : 1})

    model.fit(x_train[n:n*25,:,:,:],
            [y_train_pruned[n:n*25], x_train[n:n*25,:,:,:]],
              batch_size=batch_size,
              epochs=50,
              verbose=1)
              # validation_data=(x_test, [y_test, x_test]))

    score = model.evaluate(x_test, [y_test, x_test], verbose=0)
    for metric_name, value in zip(model.metrics_names, score):
        print(metric_name + ":", value)

    model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                  optimizer=Adam(clipnorm = 1.),
                  metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                  loss_weights={'class' : 1, 'reconstruction' : 0.0001})

    model.fit(x_train[0:n,:,:,:],
            [y_train_pruned[0:n], x_train[0:n,:,:,:]],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
              # validation_data=(x_test, [y_test, x_test]))

    predictions = model.predict(x_test)[0]
    pred_class = [np.argmax(p) for p in predictions]
    true_class = [np.argmax(p) for p in y_test]

    print(classification_report(true_class, pred_class))

    score = model.evaluate(x_test, [y_test, x_test], verbose=0)
    for metric_name, value in zip(model.metrics_names, score):
        print(metric_name + ":", value)

    example_output = model.predict(x_train[3:4,:,:,:])
    show_image(x_train[3])
    show_image(example_output[1])

def train_basic_model(n_samples_train):
    model = create_model(False)
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

    predictions = model.predict(x_test)
    pred_class = [np.argmax(p) for p in predictions]
    true_class = [np.argmax(p) for p in y_test]

    print(classification_report(true_class, pred_class))

    score = model.evaluate(x_test, y_test, verbose=0)
    for metric_name, value in zip(model.metrics_names, score):
        print(metric_name + ":", value)


# x_test = x_test[:10000,:,:,:]
# y_test = y_test[:10000,:]

train_basic_model(30)
train_regularized_model(30)

