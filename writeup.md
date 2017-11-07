## Project: Deep Learning - Follow Me

---

[//]: # (Image References)

[loss_plot]: loss_plot.PNG



## Network Architecture

We decided to use a fully convolutional network for the semantic segmentation task at hand.
This means using a regular convolutional network for the image classification, but replacing the final fully connected layers with transposed convoultional layers.
The advantage over a fully connected output layer is that the FCN is able to preserve the spatial information of the input image.
This allows us not only to determine if there is a target object in the image, but also where in the image the target
is located.

The encoder part of the network consists of several pooling layers and is responsible for extracting features from the image.
The output is not a simple (one hot) classification, but a pixel-wise classification (i.e. for each pixel, decide whether it is target, human, or background): so we need a classification output that is the same x and y shape as the input.
The output from the encoder layers is feed through several transposed convolutional layers (the decoder layer),
whose task is to upsample the image back to the original size.
Skip connections between the encoder layer and the decoder layers allow the network to use the higher resolution information
from the earlier layers to make more precise segmentation decisions.

Specifically the encoder consists of 3 seperable convoultional layers (.....), each followed by a RELU activation and a 
batch normalization step (allows the network to generalize better by making the inputs to the next layer more homegeneous).


A detailed description of the network is given below:


| Encoder Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x160x3 RGB image   							|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 80x80x3  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 80x80x32 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 40x40x32  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 40x40x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 2x2 stride, same padding, outputs 40x40x16  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 20x20x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 20x20x32 	|
| RELU					|												|
| BatchNormalization   	|  	|



| Decoder Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Upsample 2x2     	| outputs 40x40x32  	|
| Concatenate     	| Skip connection from 2nd encoder layer, outputs 40x40x48 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 40x40x48  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 40x40x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 40x40x16  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 40x40x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Upsample 2x2     	| outputs 80x80x16  	|
| Concatenate     	| Skip connection from 1st encoder layer, outputs 80x80x48 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 80x80x48  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 80x80x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 80x80x16  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 80x80x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Upsample 2x2     	| outputs 160x160x16  	|
| Concatenate     	| Skip connection from input layer, outputs 160x160x19 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 160x160x19  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 160x160x32 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 160x160x32  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 160x160x32 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 160x160x3  	|
| Softmax					|												|



To connect encoder and decoder we used a 1x1 convolution. The 1x1 convolution reduces the filter depth while preserving the coordinate (x,y) information. The advantage over a fully connected layer (where each input neuron
is connected to each output neuron) is a hugh reduction in the number of parameters.
Finally a 1x1 convolution layer is used to connect the final hidden layer to the output with shape x, y like the input image and depth equal to the number of possible classes (3 in this case).

The provided training and validation data was augmented with our own collection of preprocessed images (1094 additional images for training and 107 additional images for validation).
The final score (IoU * weight) of the model was ~48%.

<!-- ![alt text][loss_plot] -->


#### Hyper parameters

- Epoch: The final model was trained for 153 epochs, at which point training was interrupted because the network already performed reasonably well on the validation data.
Generally training for more epochs would reduce the training loss, but not necesaryly also the validation loss (if overfitting occurs).

- Learning Rate: The learning rate was set to 0.01 from experience; slightly higher and lower rates were tested, but the initial guess provided the best tradeoff between performance/velocity of convergence.
The learning rate determines how fast the weights are updateded with the gradient during the back-propagation. A high learning rate means a large change to the weights,
a small learning rate means a small change. If the learning rate is to large the network can't converge on a local optimum, if it's too small the training takes too long.

- Batch Size: The batch size was chosen according to the maximum size that could comfortably fit into memory during the learning process, in our case 128.
Batch size determines how many input images are propagated through the network at once. Using larger batches allows for more efficient computation (less overhead, e.g. device-host copy operations).
Computing the gradient over a larger batch will give a better estimate of the gradient for the whole dataset. The batch still shouldn't be too large, because some noise in the gradient estimation
is helpful in avoiding local optima.



#### 1 by 1 convolutions

The output of a convolutional layer is a 4D tensor (matrix of shape: batch_size * image_x * image_y * number_of_filters).
To connect this to a fully connected layer we have to flatten the 4D tensor to a 2D tensor (shape: batch_size * number_of_features),
thus losing the spatial information of the convolutional layer.
By using a 1 by 1 convolution instead of a fully connected layer, we can preserve the spatial information: the shape remains
the same as of the 4D tensor of the previous convolutional layer, except for the filter depth (shape: batch_size * image_x * image_y * new_number_of_filters).
Typically the 1 by 1 convolution is used to reduce the filter depth of a convolutional layer.

The 1 by 1 convolution should be used when preserving spatial information is imported, e.g. for image segmentation.
Fully connected layers are useful when we need to connect the output of a convolutional layer to a layer without spatial information,
e.g. softmax activations for image classification.



#### Encoding / decoding images

An encoder - decoder network architecture is useful for classification tasks, in which the spatial information of the original image should be preserved, e.g. for image segmentation.
Encoding or downsampling an image (e.g. by using convolutions or pooling layers) lets the network focus on individual aspects of an image:
the network could learn simple shapes like lines or curves in earlier layers and more complex features like faces in later layers.
Now for a segmentation task the output mask must be the same size as the input image. To restore the downsampled images
back to the original size of the input we can use decoders or upsampling (e.g. transposed convolutions).
The upsampled image will lack some detailed information, which was lost during the downsampling step, where we discarded the high level
image shapes to focus more on the smaller details; this can be mitigated
by using skip connections, which allow to directly feed the high resolution images from early layers into the low detail upsampled images
in later layers.


#### Following another object

The model would probably work equally well for identifing different objects, like dogs or cars, but would need to be (at least partially) retrained with different training data (target masks).
It might be possible to reuse the weights of the first layers (i.e. the general border and shape recognition layers), but the detailed features would certainly be different.


#### Future Improvements

To achieve a higher accuracy it would be beneficial to collect more training data. Increasing the filter depth and adding more hidden layers can also help top improve accuracy,
but comes at the cost of a higher latency (higher compute time). For the tracking to work well, the predictions must be fast, so the model can't be too big.
A lower learning rate can help find a better solution, but would require longer training (more epochs).

