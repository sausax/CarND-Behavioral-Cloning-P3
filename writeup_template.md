#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```<span></span>h
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 3 and 64 (model.py lines 49-68) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 48). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 53, 57...). All convolutional droput layer are set at 0.25 dropout rate and all fully connected droput layers are set at 0.5 dropout rate. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 18-23). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving in reverse direction, driving at the side of road and driving at very slow speeds.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to have enough layers to learn representation of dataset.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because it has sufficient number of convolutional layers and fully connected layers to classify images.

I splitted the dataset into training and validation set to test the performance of the model. The training process not converging, so I changed the optimizer from Adadelta to Adam.

After that I found out that the model training error is going down by each epoch, but the validation error starts increases after few epochs. This happes when model starts to overfit the training data.
To combat the overfitting, I modified the model and added dropout layers.

After that I made changes to the model by adding more convolutional layers, adding more convolutional layers. I also tried grayscale images instead of RGB images for training, but it didn't improve the error rates.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I collected more data by driving in different variations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is based on NVIDIA model for end to end learning. I added dropout and max pooling layer to it.
It (model.py lines 19-79) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x80x3 RGB image   					        | 
| Lambda         		| Normalization layer   					    |
| Convolution1     	    | kernel size: 3x3, Depth: 3	                |
| ELU					|												|
| Convolution2     	    | kernel size: 3x3, Depth: 24	                |
| ELU					|												|
| Max pooling	      	| pool size: 2x2  	                            |
| Dropout	      	    | rate: 0.25	                                |
| Convolution2     	    | kernel size: 3x3, Depth: 36	                |
| ELU					|												|
| Max pooling	      	| pool size: 2x2  	                            |
| Dropout	      	    | rate: 0.25	                                |
| Convolution2     	    | kernel size: 3x3, Depth: 48	                |
| ELU					|												|
| Max pooling	      	| pool size: 2x2  	                            |
| Dropout	      	    | rate: 0.25	                                |
| Convolution2     	    | kernel size: 3x3, Depth 64	                |
| ELU					|												|
| Max pooling	      	| pool size: 2x2  	                            |
| Dropout	      	    | rate: 0.25	                                |
| Fully connected1		| size: 1164       				                |
| Dropout				| rate: 0.50									|
| Fully connected1		| size: 100       				                |
| Dropout				| rate: 0.50									|
| Fully connected1		| size: 50       				                |
| Dropout				| rate: 0.50									|
| Fully connected1		| size: 10       				                |
| Dropout      		    | rate: .50       				                |
| Fully connected1		| size: 1       				                |


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
