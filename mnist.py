import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train)

def show_train_image(image_index):
    print(y_train[image_index])
    plt.imshow(x_train[image_index], cmap='Greys')

def show_test_image(image_index):
    print(y_test[image_index])
    plt.imshow(x_test[image_index], cmap='Greys')

def reshape_data():
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


show_test_image(8)
