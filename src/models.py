import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam, SGD, Nadam
from keras import backend as K


def simple_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='softmax'))


	model.compile(loss='mse',
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

	return model

## using different optimizer
def simple_model2(input_shape):
	lrate = 0.0001

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='tanh'))


	model.compile(loss='mse',
	              optimizer=Adam(lr=lrate))

	return model

## Using different network and optimizer
def simple_model3(input_shape):
	lrate = 0.001

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='tanh'))


	model.compile(loss='mse',
	              optimizer=Adam(lr=lrate))

	return model

## Using different network and optimizer
def simple_model5(input_shape):
	lrate = 0.0001

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='tanh'))


	model.compile(loss='mean_squared_error', optimizer=Nadam(lr=lrate))

	return model


## Using different network and optimizer
## Removed max MaxPooling2D layer
def simple_model4(input_shape):
	lrate = 0.0001

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	#model.add(Dropout(0.5))
	#model.add(Dense(64, activation='relu'))
	#model.add(Dropout(0.5))
	#model.add(Dense(32, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='tanh'))


	model.compile(loss='mse',
	              optimizer=Adam(lr=lrate))

	return model


## Using different network and optimizer
def simple_model6(input_shape):
	lrate = 0.0001

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='tanh'))


	model.compile(loss='mean_squared_error', optimizer=Adam(lr=lrate))

	return model

## Using different network and optimizer
def simple_model7(input_shape):
	lrate = 0.00005

	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='elu'))
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, (3, 3), activation='elu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, (3, 3), activation='elu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(512, (3, 3), activation='elu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1024, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(128, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='tanh'))


	model.compile(loss='mean_squared_error', optimizer=Adam())

	return model

## Using different network and optimizer
## NVIDIA arch
def simple_model8(input_shape):
	lrate = 0.00005

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
	#model.add(MaxPooling2D(pool_size=(2, 2)))
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

	return model
