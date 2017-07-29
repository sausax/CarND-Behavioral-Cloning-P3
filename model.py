import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler

from preprocessing import *

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
input_shape = (80, 80, 3)


## NVIDIA architecture
## Added max pooling and dropout layer
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
model.add(Conv2D(3, kernel_size=(3, 3),
                 activation='elu'))
model.add(Conv2D(24, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(36, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(48, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1164, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error', optimizer=Adam())

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

# Save model
model.save('model.h5')
