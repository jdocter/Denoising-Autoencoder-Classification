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
from keras.models import model_from_json
from keras_tqdm import TQDMCallback

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(image.reshape(28, 28), cmap='Greys')
    plt.show()

def save_results(description, n_labeled, class_acc, n_unlabeled=None, reg_acc = None):
    with open("results.csv",'a') as results:
        results.write("{}, {}, {}, {}, {}\n".format(description,n_labeled, class_acc, n_unlabeled, reg_acc))


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/" +name+ "_arch.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/" +name +"_weights.h5")
    print("Saved model to disk")


def load_model(name):
    # load json and create model
    json_file = open("models/" + name + "_arch.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/" +name + "_weights.h5")
    print("Loaded model from disk")
    return loaded_model



class ASL:
    """
        Autoencoder supervised learning class
        initialize class with number of labeled and unlabeled
        must call cnn_setup() or simple_setup() before any training
    """

    def __init__(self, n_samples_train_labeled, n_samples_train_unlabeled, verbose=0):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.n_labeled = n_samples_train_labeled
        self.n_unlabeled = n_samples_train_unlabeled
        self.verbose = verbose

    def cnn_setup(self):
        # Reshaping the array to 4-dims so that it can work with the Keras API
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)[:, :, :, :]
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        self.input_shape = (28, 28, 1)
        # Making sure that the values are float so that we can get decimal points after division
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value.
        self.x_train /= 255
        self.x_test /= 255

        self.input_shape = self.x_train[0].shape
        self.num_classes = 10
        self.batch_size = 1
        self.epochs = 50

        np.random.seed(42)
        self.my_init = initializers.glorot_uniform(seed=42)

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)


        # train on n labeled data, rest unlabeled
        y_train_pruned = np.copy(self.y_train)
        y_train_pruned[self.n_labeled:,:] = 0

        self.x_train_labeled = self.x_train[0:self.n_labeled,:,:,:]
        self.y_train_pruned_labeled = y_train_pruned[0:self.n_labeled]
        self.y_train_labeled = self.y_train[0:self.n_labeled,:]

        self.x_train_unlabeled = self.x_train[self.n_labeled: self.n_labeled + self.n_unlabeled,:,:,:]
        self.y_train_pruned_unlabeled = y_train_pruned[self.n_labeled: self.n_labeled + self.n_unlabeled]

        self.model_creator = self.create_cnn_model



    def simple_setup(self):
        """ data setup for simple single later autoencoder """

        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.

        # flatten
        self.x_train = self.x_train.reshape((len(self.x_train), np.prod(self.x_train.shape[1:])))
        self.x_test = self.x_test.reshape((len(self.x_test), np.prod(self.x_test.shape[1:])))

        self.input_shape = self.x_train[0].shape
        self.num_classes = 10
        self.batch_size = 20
        self.epochs = 10000

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        self.model_creator = self.create_simple_model

        # train on n labeled data, rest unlabeled
        y_train_pruned = np.copy(self.y_train)
        y_train_pruned[self.n_labeled:,:] = 0

        self.x_train_labeled = self.x_train[0:self.n_labeled,:]
        self.y_train_pruned_labeled = y_train_pruned[0:self.n_labeled]
        self.y_train_labeled = self.y_train[0:self.n_labeled,:]

        self.x_train_unlabeled = self.x_train[self.n_labeled: self.n_labeled + self.n_unlabeled,:]
        self.y_train_pruned_unlabeled = y_train_pruned[self.n_labeled: self.n_labeled + self.n_unlabeled]


        self.model_creator = self.create_simple_model


    def create_cnn_model(self, regularized):
        visible = Input(shape=self.input_shape)

        encode = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=self.my_init)(visible)
        encode = MaxPooling2D(pool_size=(2, 2))(encode)
        # encode = Dropout(0.25)(encode)
        encode = Flatten()(encode)
        encode = Dense(64, activation='relu')(encode)
        # encode = Dropout(0.25)(encode)

        output = Dense(self.num_classes, name='class', activation='softmax')(encode)

        decode = Dense(1568, activation='relu')(encode)
        decode = Reshape((14, 14, 8))(decode)
        # decode = Dropout(0.25)(decode)
        decode = UpSampling2D(size=(2, 2))(decode)
        decode = Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same', name='reconstruction', kernel_initializer=self.my_init)(decode)

        if regularized:
            return Model(inputs=visible, outputs=[output, decode])
        else:
            return Model(inputs=visible, outputs=output)

    def create_simple_model(self, regularized): # add a Dense layer with a L1 activity regularizer

        visible = Input(shape=self.input_shape)
        encode = Dense(64, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(visible)

        output = Dense(self.num_classes, name='class', activation='softmax')(encode)

        decode = Dense(784, activation='sigmoid',name='reconstruction')(encode)

        if regularized:
            return Model(inputs=visible, outputs=[output, decode])
        else:
            return Model(inputs=visible, outputs=output)


    def conditional_categorical_crossentropy(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        # this loss functions gives zero loss when there is no label
        # otherwise, categorical_crossentropy
        return K.switch(K.flatten(K.equal(K.sum(y_true, axis=-1), 0.)), K.zeros_like(loss), loss)

    def zero_loss(self, y_true, y_pred):
        return y_pred * 0

    def train_regularized_model(self):
        model = self.model_creator(True)
        model.summary()


        model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                      optimizer=Adam(clipnorm = 1.),
                      metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                      loss_weights={'class' : 0, 'reconstruction' : 1})

        model.fit(np.copy(self.x_train_unlabeled),
                [np.copy(self.y_train_pruned_unlabeled), np.copy(self.x_train_unlabeled)],
                  batch_size=self.batch_size,
                  epochs=50,
                  verbose=self.verbose,
                  callbacks=[TQDMCallback()])
                  # validation_data=(x_test, [y_test, x_test]))

        print("regularized model pure reconstruction: " + str(self.n_unlabeled) + " samples")
        score = model.evaluate(self.x_test, [self.y_test, self.x_test], verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                      optimizer=Adam(clipnorm = 1.),
                      metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                      loss_weights={'class' : 1, 'reconstruction' : 0.0001})

        model.fit(np.copy(self.x_train_labeled),
                [np.copy(self.y_train_pruned_labeled), np.copy(self.x_train_labeled)],
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=self.verbose,
                  callbacks=[TQDMCallback()])
                  # validation_data=(x_test, [y_test, x_test]))

        predictions = model.predict(self.x_test)[0]
        pred_class = [np.argmax(p) for p in predictions]
        true_class = [np.argmax(p) for p in self.y_test]

        print("\n\n==========================================================")
        print("regularized model with labels: " + str(self.n_unlabeled) + " unlabeled " + str(self.n_labeled) + " labeled")
        print(classification_report(true_class, pred_class))

        score = model.evaluate(self.x_test, [self.y_test, self.x_test], verbose=self.verbose)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        example_output = model.predict(self.x_train[3:4])
        show_image(self.x_train[3])
        show_image(example_output[1])

        save_model(model,"regularized_"+str(self.n_labeled)+"labels")
        save_results("regularized",self.n_labeled, score[3], self.n_unlabeled, score[4])


    def train_basic_model(self):
        model = self.model_creator(False)
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(np.copy(self.x_train_labeled),
                  np.copy(self.y_train_labeled),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=self.verbose,
                  callbacks=[TQDMCallback()])
                  # validation_data=(x_test, y_test))

        predictions = model.predict(self.x_test)
        pred_class = [np.argmax(p) for p in predictions]
        true_class = [np.argmax(p) for p in self.y_test]


        print("\n\n==========================================================")
        print("basic model: " + str(self.n_labeled) + " samples")
        print(classification_report(true_class, pred_class))

        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        save_model(model,"unregularized_"+str(self.n_labeled) +"labels")
        save_results("unregularized",self.n_labeled, score[1])

    def train_autoencoder(self):
        """ trains autoencoder only for entire training set """

        model = self.model_creator(True)
        model.summary()


        model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'binary_crossentropy'},
                      optimizer=Adam(clipnorm = 1.),
                      metrics={'class' : 'accuracy', 'reconstruction' : 'accuracy'},
                      loss_weights={'class' : 0, 'reconstruction' : 1})

        model.fit(self.x_train,
                [self.y_train, self.x_train],
                  batch_size=256,
                  epochs=50,
                  verbose=self.verbose,
                  callbacks=[TQDMCallback()])
                  # validation_data=(x_test, [y_test, x_test]))

        print("autoencoder: 60000 samples")
        score = model.evaluate(self.x_test, [self.y_test, self.x_test], verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        save_model(model,"autoencoder_60000")

    # x_test = x_test[:10000,:,:,:]
    # y_test = y_test[:10000,:]

asl = ASL(100,500)
asl.simple_setup()
# asl.train_basic_model()
asl.train_regularized_model()
