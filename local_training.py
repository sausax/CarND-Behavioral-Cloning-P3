
# coding: utf-8

# In[1]:

import numpy as np
from scipy.misc import imread, imresize

def read_imgs(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)

    return imgs

def resize(imgs, shape=(32, 16, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)

    return imgs_resized

def rgb2gray(imgs):
    """
    Convert images to grayscale.
    """
    return np.mean(imgs, axis=3, keepdims=True)

def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return imgs / (255.0 / 2) - 1

def preprocess(imgs):
    imgs_processed = resize(imgs)
    imgs_processed = rgb2gray(imgs_processed)
    imgs_processed = normalize(imgs_processed)

    return imgs_processed

def random_flip(imgs, angles):
    """
    Augment the data by randomly flipping some angles / images horizontally.
    """
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles

def augment(imgs, angles):
    imgs_augmented, angles_augmented = random_flip(imgs, angles)

    return imgs_augmented, angles_augmented

def gen_batches(imgs, angles, batch_size):
    """
    Generates random batches of the input data.
    :param imgs: The input images.
    :param angles: The steering angles associated with each image.
    :param batch_size: The size of each minibatch.
    :yield: A tuple (images, angles), where both images and angles have batch_size elements.
    """
    num_elts = len(imgs)

    while True:
        indeces = np.random.choice(num_elts, batch_size)
        batch_imgs_raw, angles_raw = read_imgs(imgs[indeces]), angles[indeces].astype(float)

        batch_imgs, batch_angles = augment(preprocess(batch_imgs_raw), angles_raw)

        yield batch_imgs, batch_angles


# In[15]:

# Import libraries necessary for this project.
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import csv

from keras.models import Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, ELU, Flatten
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam

from PIL import Image

from sklearn.model_selection import train_test_split

from IPython.display import display

#get_ipython().magic(u'matplotlib inline')


# In[16]:

# Location of the simulator data.
DATA_FILE = '/home/saurabh/Downloads/car_output/driving_log.csv'

# Load the training data from the simulator.
cols = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
data = pd.read_csv(DATA_FILE, names=cols, header=1)


# In[17]:

#print data['center_image'][0]


# In[22]:

flags = tf.app.flags
FLAGS = flags.FLAGS

#flags.DEFINE_string('imgs_dir', '/Users/saurabh/Downloads/car_output/IMG/', 'The directory of the image data.')
#flags.DEFINE_string('data_path', DATA_FILE, 'The path to the csv of training data.')
#flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
#flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to train for.')
#flags.DEFINE_float('lrate', 0.0001, 'The learning rate for training.')
batch_size =128
num_epochs = 10
lrate = 0.0001


def main():
    ##
    # Load Data
    ##

    # with open(DATA_FILE, 'r') as f:
    #     reader = csv.reader(f)
    #     # data is a list of tuples (img path, steering angle)
    #     data = np.array([row for row in reader])

    # # Split train and validation data
    # np.random.shuffle(data)
    # split_i = int(len(data) * 0.9)
    # X_train, y_train = list(zip(*data[:split_i]))
    # X_val, y_val = list(zip(*data[split_i:]))

    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_val, y_val = np.array(X_val), np.array(y_val)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X = np.array(data['center_image'])
    y = np.array(data['steering_angle'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    ##
    # Define Model
    ##

    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        Conv2D(64, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Conv2D(128, 3, 3, border_mode='same', activation='relu'),
        Conv2D(256, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    model.compile(optimizer=Adam(lr=lrate), loss='mse')

    ##
    # Train
    ##

    history = model.fit_generator(gen_batches(X_train, y_train, batch_size),
                                  len(X_train),
                                  num_epochs,
                                  validation_data=gen_batches(X_test, y_test, batch_size),
                                  nb_val_samples=len(X_test))

    ##
    # Save model
    ##

    json = model.to_json()
    model.save_weights('save/model.h5')
    with open('save/model.json', 'w') as f:
        f.write(json)



main()

