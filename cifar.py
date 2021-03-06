import keras
import numpy as np
from keras import initializers
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras import regularizers
from keras import backend as K
from keras.models import model_from_json

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import json

def show_image(image):
    plt.figure()
    plt.imshow(image.reshape(32, 32, 3), cmap='Greys')

def save_quad_image(image, file_name):
    plt.imsave(file_name, image.reshape(128, 32, 3))

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
    print(">>>>>>>> Saved model to disk as " + name)

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

def save_model_with_history(model, history_dict, name):
    save_model(model, name)
    json.dump(history_dict, open("models/" + name + "_history.json", 'w'))

def load_history(name):
    return json.load(open("models/" + name + "_history.json"))

def add_noise(x_list, noise_type):
    # mask over random pixels
    if noise_type == 's&p':
        x_list *= np.stack([np.random.uniform(size=(x_list.shape[:-1])) < 0.9]*3, axis=3)
        np.maximum(x_list, np.stack([np.random.uniform(size=(x_list.shape[:-1])) < 0.1]*3, axis=3) * 1.0, x_list)
    elif noise_type == 'blackout':
        x_list *= np.stack([np.random.uniform(size=(x_list.shape[:-1])) < 0.3]*3, axis=3)


def plot_history(history, class_acc_ax=None, rec_loss_fig=None):
    if class_acc_ax is not None:
        if "class_acc" in history.keys():
            class_acc_ax.plot(history['class_acc'], color='C0', linestyle='-', label='train_acc_regularized')
            class_acc_ax.plot(history['val_class_acc'], color='C0', linestyle='--', label='val_acc_regularized')
            print("Regularized")
            print(history['val_class_acc'][-1] * 100, np.std(history['val_class_acc'][-5:]) * 100)
        else:
            class_acc_ax.plot(history['acc'], color='C1',linestyle='-', label='train_acc_unregularized')
            class_acc_ax.plot(history['val_acc'], color='C1',linestyle='--', label='val_acc_unregularized')
            print("Unregularized")
            print(history['val_acc'][-1] * 100, np.std(history['val_acc'][-5:]) * 100)

    if rec_loss_fig is not None:
        plt.figure(rec_loss_fig.number)
        plt.plot(history['reconstruction_mean_squared_error'],
                color='C0', linestyle='-', label='train_rec_loss')
        plt.plot(history['val_reconstruction_mean_squared_error'],
                color='C0', linestyle='--', label='val_rec_loss')

        # print(history['reconstruction_mean_squared_error'][-1])
        # print(history['val_reconstruction_mean_squared_error'][-1])

class ASL:
    """
        Autoencoder supervised learning class
        initialize class with number of labeled and unlabeled
        must call cnn_setup() or simple_setup() before any training
    """

    def __init__(self, n_samples_train_labeled, noise_type, verbose=1):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # Set constants
        self.n_labeled = n_samples_train_labeled
        self.verbose = verbose
        self.noise_type = noise_type
        self.num_classes = 10

        # Normalize pixel values
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.

        # Add noise to inputs
        self.x_train_noisy = np.copy(self.x_train)
        self.x_test_noisy = np.copy(self.x_test)
        add_noise(self.x_train_noisy, self.noise_type)
        add_noise(self.x_test_noisy, self.noise_type)

        # print("Displaying example of input with/without noise, close to continue ... ")
        # show_image(self.x_train[6])
        # show_image(self.x_train_noisy[6])

        # Flatten
        self.x_train = self.x_train.reshape((len(self.x_train), np.prod(self.x_train.shape[1:])))
        self.x_test = self.x_test.reshape((len(self.x_test), np.prod(self.x_test.shape[1:])))
        self.x_train_noisy = self.x_train_noisy.reshape((len(self.x_train_noisy), np.prod(self.x_train_noisy.shape[1:])))
        self.x_test_noisy = self.x_test_noisy.reshape((len(self.x_test_noisy), np.prod(self.x_test_noisy.shape[1:])))

        # Convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        # Zero out labels on unlabeled data to be rigorous
        self.y_train[self.n_labeled:,] = 0.0

    def cnn_setup(self):
        """ data setup for cnn autoencoder """

        # 3D CNN input
        self.x_train = self.x_train.reshape((len(self.x_train), 32, 32, 3))
        self.x_test = self.x_test.reshape((len(self.x_test), 32, 32, 3))
        self.x_train_noisy = self.x_train_noisy.reshape((len(self.x_train_noisy), 32, 32, 3))
        self.x_test_noisy = self.x_test_noisy.reshape((len(self.x_test_noisy), 32, 32, 3))

        # Create subset for labeled data only
        self.x_train_labeled = self.x_train[0:self.n_labeled,:]
        self.x_train_noisy_labeled = self.x_train_noisy[0:self.n_labeled,:]
        self.y_train_labeled = self.y_train[0:self.n_labeled,:]

        self.input_shape = self.x_train[0].shape
        self.model_creator = self.create_cnn_model
        self.architecture = "cnn"

    def simple_setup(self):
        """ data setup for dense models """

        # Data should already be flattened
        # Create subset for labeled data only
        self.x_train_labeled = self.x_train[0:self.n_labeled,:]
        self.x_train_noisy_labeled = self.x_train_noisy[0:self.n_labeled,:]
        self.y_train_labeled = self.y_train[0:self.n_labeled,:]
        
        self.input_shape = self.x_train[0].shape
        self.model_creator = self.create_simple_model
        self.architecture = "dense"

    def create_cnn_model(self, model_type):
        visible = Input(shape=self.input_shape)

        encode = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(visible)
        encode = BatchNormalization()(encode)
        encode = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(encode)
        encode = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(encode)
        encode = BatchNormalization()(encode)
        encode = Flatten()(encode)

        output = Dense(self.num_classes, name='class', activation='softmax')(encode)

        decode = Reshape((16, 16, 32))(encode)
        decode = UpSampling2D()(decode)
        decode = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(decode)
        decode = BatchNormalization()(decode)
        decode = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid', name="reconstruction")(decode)

        if model_type == "dual":
            return Model(inputs=visible, outputs=[output, decode])
        elif model_type == "basic":
            return Model(inputs=visible, outputs=output)
        elif model_type == "autoencoder":
            return Model(inputs=visible, outputs=decode)

    def create_simple_model(self, model_type): # add a Dense layer with a L1 activity regularizer

        visible = Input(shape=self.input_shape)
        # encode = Dropout(0.2)(visible)
        encode = Dense(1024, activation='relu')(visible)

        output = Dense(self.num_classes, name='class', activation='softmax')(encode)

        decode = Dropout(0.2)(encode)
        decode = Dense(3072, activation='sigmoid',name='reconstruction')(decode)

        if model_type == "dual":
            return Model(inputs=visible, outputs=[output, decode])
        elif model_type == "basic":
            return Model(inputs=visible, outputs=output)
        elif model_type == "autoencoder":
            return Model(inputs=visible, outputs=decode)


    def conditional_categorical_crossentropy(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        # this loss functions gives zero loss when there is no label
        # otherwise, categorical_crossentropy
        return K.switch(K.flatten(K.equal(K.sum(y_true, axis=-1), 0.)), K.zeros_like(loss), loss)

    def zero_loss(self, y_true, y_pred):
        return y_pred * 0

    def train_regularized_model(self, lr=0.00005, batch_size=16, epochs=40, loss_weights=[1,1]):
        # Assume a denoise autoencoder has been trained
        print("Loading most recently trained de-noising auto-encoder ... ")
        model = load_model("denoise_autoencoder_most_recent")

        model.compile(loss={'class' : 'categorical_crossentropy', 'reconstruction' : 'mse'},
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics={'class' : 'accuracy', 'reconstruction' : 'mse'},
                      loss_weights={'class' : loss_weights[0], 'reconstruction' : loss_weights[1]})

        history = model.fit(np.copy(self.x_train_noisy_labeled),
                  [np.copy(self.y_train_labeled), np.copy(self.x_train_labeled)],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=self.verbose,
                  validation_data=(self.x_test_noisy, [self.y_test, self.x_test]))

        predictions = model.predict(self.x_test_noisy)[0]
        pred_class = [np.argmax(p) for p in predictions]
        true_class = [np.argmax(p) for p in self.y_test]

        print("\n\n==========================================================")
        print("regularized model with " + str(self.n_labeled) + " labeled samples")
        print(classification_report(true_class, pred_class))

        score = model.evaluate(self.x_test_noisy, [self.y_test, self.x_test], verbose=self.verbose)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        # example_output = model.predict(self.x_test_noisy[9:10])
        # show_image(self.x_test_noisy[9])
        # show_image(self.x_test[9])
        # show_image(example_output[1])

        save_model_with_history(model, history.history,
            str(self.noise_type) + "_" + str(self.n_labeled) + "labels_" +
            str(self.architecture) + "_regularized")
        save_results("regularized",self.n_labeled, score[3], 123123, score[4])


    def train_basic_model(self, lr=0.00005, batch_size=16, epochs=40):
        model = self.model_creator(model_type="basic")
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])

        history = model.fit(np.copy(self.x_train_noisy_labeled),
                  np.copy(self.y_train_labeled),
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=self.verbose,
                  validation_data=(self.x_test_noisy, self.y_test))
                  #callbacks=[TQDMCallback()]))

        predictions = model.predict(self.x_test_noisy)
        pred_class = [np.argmax(p) for p in predictions]
        true_class = [np.argmax(p) for p in self.y_test]

        print("\n\n==========================================================")
        print("basic model: " + str(self.n_labeled) + " samples")
        print(classification_report(true_class, pred_class))

        score = model.evaluate(self.x_test_noisy, self.y_test, verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        save_model_with_history(model, history.history,
            str(self.noise_type) + "_" + str(self.n_labeled) + "labels_" +
            str(self.architecture) + "_unregularized")
        save_results("unregularized",self.n_labeled, score[1])

    def train_denoise_autoencoder(self, lr=0.00005, batch_size=32, epochs=20):
        """ Trains an auto-encoder which removes noise from input, full training set """

        model = self.model_creator(model_type="dual")
        model.summary()

        # No loss on classification
        model.compile(loss={'class' : self.zero_loss, 'reconstruction' : 'mse'},
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics={'class' : 'accuracy', 'reconstruction' : 'mse'},
                      loss_weights={'class' : 0, 'reconstruction' : 1})

        # Training input: noisy input images
        # Training output: null class output & clean input images
        history = model.fit(self.x_train_noisy,
                  [np.zeros(self.y_train.shape), self.x_train],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=self.verbose,
                  validation_data=(self.x_test_noisy, [self.y_test, self.x_test]))

        print("=======================================================")
        print("Results for denoise autoencoder, trained on all samples")
        score = model.evaluate(self.x_test_noisy, [self.y_test, self.x_test], verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        save_model_with_history(model, history.history,
            str(self.noise_type) + "_" + str(self.n_labeled) + "labels_" +
            str(self.architecture) + "_denoise")
        save_model(model, "{}_{}labels_{}_lr{}_batch{}_epochs{}".format(
            self.noise_type, self.n_labeled, self.architecture,
            lr, batch_size, epochs))
        save_model(model,"denoise_autoencoder_most_recent".format(lr, batch_size, epochs))

        # Show example of denoising
        # example_output = model.predict(self.x_test_noisy[3:4])
        # show_image(self.x_test[3])
        # show_image(self.x_test_noisy[3])
        # show_image(example_output[1])

    def train_simplest_autoencoder(self):
        model = self.model_creator(model_type="autoencoder")
        model.summary()

        # model = load_model("no_noise_autoencoder")

        model.compile(loss=['mse'],
                      optimizer=keras.optimizers.Adam(lr=0.0001, clipnorm=1.0),
                      metrics=['mse'])

        model.fit(self.x_train,
                  self.x_train,
                  batch_size=32,
                  epochs=20,
                  verbose=self.verbose,
                  validation_data=(self.x_test[:1000,], self.x_test[:1000,]))

        score = model.evaluate(self.x_test, self.x_test, verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        save_model(model, "no_noise_autoencoder")

        example_output = model.predict(self.x_test_noisy[10:11])
        show_image(self.x_test[10])
        show_image(example_output)

    def test_dual(self, model_name):
        # Tests a model with two outputs (sometimes called with a denoiser, so no classifications)
        model = load_model(model_name)

        model.compile(loss={'class' : self.zero_loss, 'reconstruction' : 'mse'},
                      optimizer=keras.optimizers.Adam(),
                      metrics={'class' : 'accuracy', 'reconstruction' : 'mse'},
                      loss_weights={'class' : 0, 'reconstruction' : 1})

        print("<<<<<<<< Loaded " + model_name)

        predictions = model.predict(self.x_test_noisy)[0]
        pred_class = [np.argmax(p) for p in predictions]
        true_class = [np.argmax(p) for p in self.y_test]

        print("\n\n==========================================================")
        print(classification_report(true_class, pred_class))

        score = model.evaluate(self.x_test_noisy, [self.y_test, self.x_test], verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

        if "denoise" in model_name:
            print("Showing example denoising of images ... ")
            example_output = model.predict(self.x_test_noisy[10:14])
            save_quad_image(self.x_test[10:14],
                    "quad_original.png".format(self.noise_type))
            save_quad_image(self.x_test_noisy[10:14],
                    "quad_{}.png".format(self.noise_type, self.architecture))
            save_quad_image(example_output[1],
                    "quad_{}_{}_rec.png".format(self.noise_type, self.architecture))

    def test_basic(self, model_name):
        # Tests a model with two outputs (sometimes called with a denoiser, so no classifications)
        model = load_model(model_name)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        print("<<<<<<<< Loaded " + model_name)

        predictions = model.predict(self.x_test_noisy)
        pred_class = [np.argmax(p) for p in predictions]
        true_class = [np.argmax(p) for p in self.y_test]

        print("\n\n==========================================================")
        print(classification_report(true_class, pred_class))

        score = model.evaluate(self.x_test_noisy, self.y_test, verbose=0)
        for metric_name, value in zip(model.metrics_names, score):
            print(metric_name + ":", value)

    # x_test = x_test[:10000,:,:,:]
    # y_test = y_test[:10000,:]

""" DISCUSSION POINT 2: NOISE TYPE """

""" Experiment runners: use to generate models """

def salt_and_pepper_dense_run_experiment():
    """ Generates fully-connected models which demonstrate improvement for salt and pepper noise """
    for n_train_samples in (10000,):
        print("Salt and pepper dense experiment with {} labeled training samples".format(n_train_samples))
        asl = ASL(n_samples_train_labeled=n_train_samples, noise_type="s&p")
        asl.simple_setup()
        asl.train_denoise_autoencoder()
        asl.train_regularized_model()
        asl.train_basic_model()

def salt_and_pepper_cnn_run_experiment():
    """ Generates fully-connected models which demonstrate improvement for salt and pepper noise """
    for n_train_samples in (10000,):
        print("Salt and pepper dense experiment with {} labeled training samples".format(n_train_samples))
        asl = ASL(n_samples_train_labeled=n_train_samples, noise_type="s&p")
        asl.cnn_setup()
        asl.train_denoise_autoencoder()
        """ DISCUSSION POINT 3: LOSS_WEIGHTS FOR CNN EXPERIMENT """
        asl.train_regularized_model(lr=0.00001, loss_weights=[0.01,1])
        asl.train_basic_model()

def blackout_cnn_run_experiment():
    """ Generates fully-connected models which demonstrate improvement for salt and pepper noise """
    for n_train_samples in (10000,):
        print("Blackout dense experiment with {} labeled training samples".format(n_train_samples))
        asl = ASL(n_samples_train_labeled=n_train_samples, noise_type="blackout")
        asl.cnn_setup()
        asl.train_denoise_autoencoder()
        asl.train_regularized_model(lr=0.00001, loss_weights=[0.01,1])
        asl.train_basic_model()

""" Functions to visualize results """

def evaluate_models(noise_type, n_labeled, architecture):
    name_prefix = str(noise_type) + "_" +str(n_labeled) + "labels_"\
                + str(architecture) + "_"

    asl = ASL(n_samples_train_labeled=n_labeled, noise_type=noise_type)
    if architecture == "cnn":
        asl.cnn_setup()
    else:
        asl.simple_setup()

    asl.test_dual(name_prefix + "denoise")
    asl.test_dual(name_prefix + "regularized")
    asl.test_basic(name_prefix + "unregularized")

def view_result(noise_type, n_labeled, architecture, ax):
    print("{} {} experiment with {} labeled training samples".format(noise_type, architecture, n_labeled))

    name_prefix = str(noise_type) + "_" +str(n_labeled) + "labels_"\
                + str(architecture) + "_"

    plot_history(load_history(name_prefix + "regularized"), class_acc_ax=ax)
    plot_history(load_history(name_prefix + "unregularized"), class_acc_ax=ax)
    ax.set_title("{} noise\n{} labeled, {} network".format(
        noise_type, n_labeled, architecture))
    ax.set(xlabel='epoch', ylabel='accuracy')

    my_fig = plt.figure()
    plt.title("Reconstruction MSE over epochs\n{} noise, {} network".format(
        noise_type, architecture))
    plot_history(load_history(name_prefix + "denoise"), rec_loss_fig=my_fig)
    plt.legend()
    plt.savefig("{}_{}_rec_loss_graph.png".format(noise_type, architecture))

# Uncomment to re-train models
# salt_and_pepper_dense_run_experiment()
# salt_and_pepper_cnn_run_experiment()
# blackout_dense_run_experiment()
# blackout_cnn_run_experiment()

# Uncomment to print validation accuracy and reconstruction loss of models
# evaluate_models("s&p", 10000, "dense")
# evaluate_models("s&p", 10000, "cnn")
# evaluate_models("blackout", 10000, "dense")
# evaluate_models("blackout", 10000, "cnn")

# Generates learning curves of various models
def gen_figures():
    final_fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(13,13))
    plt.title("Classification accuracy over epochs")
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    row = 0
    for noise_type in ("s&p", "blackout"):
        view_result(noise_type, 50000, "dense", axes[row,0])
        view_result(noise_type, 10000, "dense", axes[row,1])
        view_result(noise_type, 1000, "dense", axes[row,2])
        row += 1
        view_result(noise_type, 50000, "cnn", axes[row,0])
        view_result(noise_type, 10000, "cnn", axes[row,1])
        view_result(noise_type, 1000, "cnn", axes[row,2])
        row += 1
    axes[0,2].legend()
    plt.figure(final_fig.number)
    plt.savefig("all_acc_graph.png")

gen_figures()

# asl = ASL(n_samples_train_labeled=1000, noise_type="s&p")
# asl.cnn_setup()
# asl.train_simplest_autoencoder()
# asl.simple_setup()
# asl.train_basic_model()
# asl.train_denoise_autoencoder()
# asl.train_regularized_model()
# asl.check_autoencoder()
