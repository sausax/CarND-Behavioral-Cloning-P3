import math

import numpy as np
import pandas as pd

from scipy.misc import imread, imresize

def convert_to_gray(img):
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return np.expand_dims(img, axis=2)


def crop_img(img):
	img = img[60:140, :]
	return img

def resize_img(img):
	img = imresize(img, (80, 80))
	return img

def flip_img(img):
	return np.fliplr(img)

def preprocess(img):
	img = crop_img(img)
	img = resize_img(img)
	#img = convert_to_gray(img)
	return img

def get_image_vector(img_file):
	img_arr = imread(img_file)
	img_arr = preprocess(img_arr)
	return img_arr

def create_dataset(driving_logs):
	img_lst = driving_logs['center_img']
	X_data = []
	for img in img_lst:
		X_data.append(get_image_vector(img))
	return np.array(X_data)

def load_dataset(driving_log_file):
	driving_log = pd.read_csv(driving_log_file)
	driving_log.columns = ['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed']
	return driving_log

def add_flipped_images(X_data, y_data):
	dataset_size = X_data.shape[0]
	X_flipped = []
	y_flipped = []
	for i in range(dataset_size):
		X_flipped.append(flip_img(X_data[i]))
		y_flipped.append(-y_data[i])
	X_new = np.array(X_flipped)
	y_new = np.array(y_flipped)
	X_data = np.concatenate([X_data, X_new])
	y_data = np.concatenate([y_data, y_new])

	return (X_data, y_data)

def randomly_remove_zero_angle_images(X_data, y_data, to_keep=0.5):
	index_without_zero = np.where(y_data != 0.0)
	X_nonzero = X_data[index_without_zero]
	y_nonzero = y_data[index_without_zero]

	index_with_zero = np.where(y_data == 0.0)[0]
	len_zero = index_with_zero.shape[0]
	to_keep_len = int(to_keep * len_zero)
	index_with_zero = np.random.permutation(index_with_zero)[:to_keep_len]

	X_zero = X_data[index_with_zero]
	y_zero = y_data[index_with_zero]

	X_data = np.concatenate([X_nonzero, X_zero])
	y_data = np.concatenate([y_nonzero, y_zero])
	return (X_data, y_data)

def combine_multiple_datasets(log_file_lst):
	X_data = None
	y_data = None
	for log_file in log_file_lst:
		driving_log = load_dataset(log_file)

		# Apply preprocessing functions

		X_data_tmp = create_dataset(driving_log)
		y_data_tmp = np.array(driving_log['steering_angle'])

		if X_data is None:
			X_data = X_data_tmp
		else:
			X_data = np.concatenate([X_data, X_data_tmp])

		if y_data is None:
			y_data = y_data_tmp
		else:
			y_data = np.concatenate([y_data, y_data_tmp])

	return (X_data, y_data)


# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.0001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate