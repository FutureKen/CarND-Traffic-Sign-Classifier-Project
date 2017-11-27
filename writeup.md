#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/1.jpg "Visualization"
[image2]: ./writeup_imgs/2.jpg "Grayscaling"
[image3]: ./writeup_imgs/3.jpg "Random Generated Data"
[image4]: ./writeup_imgs/4.jpg "Augmented Data"
[image5]: ./writeup_imgs/5.jpg "ten new images"
[image6]: ./writeup_imgs/6.jpg "Top 3 probs1"
[image7]: ./writeup_imgs/7.jpg "Top 3 probs2"
[image8]: ./writeup_imgs/8.jpg "Visualize state"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/futureken/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. I used the numpy library to calculate summary statistics of the traffic
signs data set:


* The size of training set is 34799
* The size of the validation set is 4410
* Number of testing set is 12630
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43



####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributing between each label type.

![Visualization][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the amount of variables and shortens the training time, also the journal of Pierre Sermanet and Yann LeCun mentioned that they got a increased accuracy with the grayscale training set.

Here is an example of a traffic sign image before and after grayscaling.

![Original vs Grayscale image][image2]

At second step, I decided to generate some fake data based on the given dataset.

As a last step, I normalized the image data use function  (X - 128) / 128, so that all the input variables have the same treatment in the model and the coefficients of a model are not scaled with respect to the units of the inputs.

From the dataset distribution histogram we can see that some of the classes have much less training data than others, so that the model would be biased toward other similar classes with larger training sets. For this reason, I decided to generate additional data, especially for those classes which has less than 1000 images. 

To add more data to the the data set, I used the following techniques:

* Random brightness, to cover more light conditions, e.g. daytime vs nighttime.
* Random translation, the traffic sign are not necessarily stays in the center of each image, it could be close to the edges of image.
* Random scale, it could be far away or near when capturing a traffic sign picture.
* Random warp, mimic the result taken from aside angle.

Here is an example of an original image and 7 augmented images:

![Original vs 7 random augmented data][image3]

The augmented data set shows in the following histogram:

![Augmented dataset][image4] 


####2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation									|
| Dropout				| Keep Probability 0.9							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| Activation									|
| Dropout				| Keep Probability 0.9							|
| Max pooling	      	| Input 10x10x16. Output 5x5x16.				|
| Flatten				| Input 5x5x16. Output 400 					|
| Fully connected		| Input 400. Output 120 					|
| RELU					| Activation									|
| Dropout				| Keep Probability 0.5							|
| Fully connected		| Input 120. Output 84 					|
| RELU					| Activation									|
| Dropout				| Keep Probability 0.5							|
| Softmax				|         									|
|						|												|

 
####3. Model training

To train the model, I used the LeNet model with AdamOptimizer. With Epochs of 60, batch size 200 and learning rate of 0.0008.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


I firstly used the LeNet model in the quiz of last class section. After I tuned the learning rate and epoch step by step and ran the model training with them, the validation accuracy somehow has a ceiling around 0.91, with learning rate of 0.0007, epoch of 60 and batch size of 128.

Then I tried converting the whole dataset to grayscale, normalized them. With the same parameters in the previous result, the validation accuracy increased to around 0.94.

I augmented the data to make sure each class has at least 1000 training images. With the same parameters, the validation accuracy improved a little to around 0.95. 

I found that the validation accuracy was much lower than the test accuracy, and I also noticed that some of the augmented data looks similar with the original ones. This indicates that the model is over fitting. So I added a dropout layer after each fully connected layer. The validation accuracy raised to around 0.96. I tried adding dropout layer after each convulutional layer, with a higher keep probability. The accuracy stopped at around 0.97 after I tuned with the Epoch, Batchsize, learning rate and keep probability parameters.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.969
* test set accuracy of 0.944

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![new test image][image5]

The third image might be difficult to classify because there are multiple similar classes which looks alike it.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Pedestrians                     | Pedestrians|
|Roundabout mandatory            | Roundabout mandatory|
|Road work                       | Road work |
|Priority road                   | Priority road|
|Turn right ahead                | Turn right ahead|
|No entry                        | No entry  |
|Stop                            | Stop      |
|Yield                           | Yield     |
|Speed limit (30km/h)            | Speed limit (30km/h)|
|No passing                      | No passing|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.943

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the visualization graphs of top 3 probabilities for each test image.

![Top 3 probs][image6] 
![Top 3 probs][image7] 

For all ten images, the model is pretty certain with the prediction to be what they should be, with probability over 90%. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here are some of the NN states with test images.
![States][image8] 

From the feature maps, we can see that the neural network use the shapes and lines of the traffic sign mostly.