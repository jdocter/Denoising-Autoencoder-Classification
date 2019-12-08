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
from keras import regularizers
from keras import backend as K

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def show_image(image):
    plt.imshow(image.reshape(28, 28), cmap='Greys')
    plt.show()
# ===============================================================================
# ===============================================================================
# TODO corrects reshaping for simple model!!!! should be flattened essentially...

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

def create_model_simple(regularized): # add a Dense layer with a L1 activity regularizer

    visible = Input(shape=input_shape)
    encode = Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(visible)
    flattened = Flatten()(encode)

    output = Dense(num_classes, name='class', activation='softmax')(flattened)

    decode = Dense(1, activation='sigmoid',name='reconstruction')(encode)
    print(decode.shape)
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

def train_regularized_model(n_samples_train_labeled, n_samples_train_unlabeled,model_creator,verbose=1):
    model = model_creator(True)
    model.summary()

    ln = n_samples_train_labeled
    un = n_samples_train_unlabeled

    # train model with n_samples_train labelled data, rest un-labelled
    y_train_pruned = y_train
    y_train_pruned[un:,:] = 0

    model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                  optimizer=Adam(clipnorm = 1.),
                  metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                  loss_weights={'class' : 0, 'reconstruction' : 1})

    model.fit(x_train[ln:un+ln,:,:,:],
            [y_train_pruned[ln:un+ln], x_train[ln:un+ln,:,:,:]],
              batch_size=batch_size,
              epochs=50,
              verbose=verbose)
              # validation_data=(x_test, [y_test, x_test]))

    print("regularized model pure reconstruction: " + str(un) + " samples")
    score = model.evaluate(x_test, [y_test, x_test], verbose=0)
    for metric_name, value in zip(model.metrics_names, score):
        print(metric_name + ":", value)

    model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                  optimizer=Adam(clipnorm = 1.),
                  metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                  loss_weights={'class' : 1, 'reconstruction' : 0.0001})

    model.fit(x_train[0:un,:,:,:],
            [y_train_pruned[0:un], x_train[0:un,:,:,:]],
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose)
              # validation_data=(x_test, [y_test, x_test]))

    predictions = model.predict(x_test)[0]
    pred_class = [np.argmax(p) for p in predictions]
    true_class = [np.argmax(p) for p in y_test]
    
    print("regularized model with labels: " + str(un) + " unlabeled " + str(ln) + " labeled")
    print(classification_report(true_class, pred_class))

    score = model.evaluate(x_test, [y_test, x_test], verbose=verbose)
    for metric_name, value in zip(model.metrics_names, score):
        print(metric_name + ":", value)

    example_output = model.predict(x_train[3:4,:,:,:])
    show_image(x_train[3])
    show_image(example_output[1])

def train_basic_model(n_samples_train,model_creator,verbose=1):
    model = model_creator(False)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train[:n_samples_train,:,:,:],
              y_train[:n_samples_train,:],
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose)
              # validation_data=(x_test, y_test))

    predictions = model.predict(x_test)
    pred_class = [np.argmax(p) for p in predictions]
    true_class = [np.argmax(p) for p in y_test]

    print("basic model: " + str(n_samples_train) + " samples")
    print(classification_report(true_class, pred_class))

    score = model.evaluate(x_test, y_test, verbose=0)
    for metric_name, value in zip(model.metrics_names, score):
        print(metric_name + ":", value)


# x_test = x_test[:10000,:,:,:]
# y_test = y_test[:10000,:]


train_basic_model(30,create_model_simple,0)
train_regularized_model(30,1000,create_model_simple,0)

