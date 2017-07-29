import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import LearningRateScheduler

from preprocessing import *
from models import *

log_files = ['/home/saurabh/Datasets/behaviour_cloning/driving_log.csv', \
			'/home/saurabh/Datasets/behaviour_cloning_reverse/driving_log.csv',\
			'/home/saurabh/Datasets/behaviour_cloning_left/driving_log.csv',\
			'/home/saurabh/Datasets/behaviour_cloning_center/driving_log.csv',\
			'/home/saurabh/Datasets/behaviour_cloning_center_reverse/driving_log.csv',\
			#'/home/saurabh/Datasets/behaviour_cloning_right/driving_log.csv',\
			'/home/saurabh/Datasets/behaviour_cloning2/driving_log.csv'] 
# Load dataset

X_data, y_data = combine_multiple_datasets(log_files)
X_data, y_data = randomly_remove_zero_angle_images(X_data, y_data, 0.25)

X_data, y_data = add_flipped_images(X_data, y_data)

# Train classifier

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.20, random_state=42)

print("Shape of X_train: ", X_train.shape)
print("Shape of y_train: ", y_train.shape)

#batch_size = 128
batch_size = 128
epochs = 30
#input_shape = (80, 80, 1)
input_shape = (80, 80, 3)
#input_shape = (80, 320, 3)

# Model with 3 convolutional layers followed by 2 fully 
# connected layer


## Train Classifier

model = simple_model8(input_shape)

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks_list)

#K.clear_session()

# Save model
model.save('trained_model.h5')
