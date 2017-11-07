## Project: Deep Learning - Follow Me

---

[//]: # (Image References)

[loss_plot]: loss_plot.PNG



## Network Architecture

Our final model consisted of the following layers:


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
| Concatenate     	| outputs 40x40x48 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 40x40x48  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 40x40x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 40x40x16  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 40x40x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Upsample 2x2     	| outputs 80x80x16  	|
| Concatenate     	| outputs 80x80x48 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 80x80x48  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 80x80x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 80x80x16  	|
| Convolution 1x1     	| 1x1 stride, same  padding, outputs 80x80x16 	|
| RELU					|												|
| BatchNormalization   	|  	|
| Upsample 2x2     	| outputs 160x160x16  	|
| Concatenate     	| outputs 160x160x19 	|
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


For the image classification we decided on an Encoder-Decoder Network.

The encoder part of the network consists of several pooling layers and is responsible for extracting features from the image.
The output is not a simple (one hot) classification, but a pixel-wise classification (i.e. for each pixel, decide whether it is target, human, or background): so we need a classification output that is the same x and y shape as the input.
To map the feature maps from the encoder layer to the output layer, we have to upsample the result of each encoder layer to restore the original shape. This happens in the decoder part.
To connect encoder and decoder we used a 1x1 convolution. The 1x1 convolution reduces the filter depth while preserving the coordinate (x,y) information. The advantage over a fully connected layer (where each input neuron
is connected to each output neuron) is a hugh reduction in the number of parameters.
Finally a 1x1 convolution layer is used to connect the final hidden layer to the output with shape x, y like the input image and depth equal to the number of possible classes (3 in this case).

The provided training and validation data was augmented with our own collection of preprocessed images (1094 additional images for training and 107 additional images for validation).
The final score (IoU * weight) of the model was ~48%.

<!-- ![alt text][loss_plot] -->


#### Hyper parameters
The batch size was chosen according to the maximum size that could comfortably fit into memory during the learning process, in our case 128.
The learning rate was set to 0.01 from experience; slightly higher and lower rates were tested, but the initial guess provided the best tradeoff between performance/velocity of convergence.
The final model was trained for 153 epochs, at which point training was interrupted because the network already performed reasonably well on the validation data.


#### Following another object
The model would probably work equally well for identifing different objects, like dogs or cars, but would need to be (at least partially) retrained with different training data (target masks).
It might be possible to reuse the weights of the first layers (i.e. the general border and shape recognition layers), but the detailed features would certainly be different.


#### Future Improvements
To achieve a higher accuracy it would be beneficial to collect more training data. Increasing the filter depth and adding more hidden layers can also help top improve accuracy,
but comes at the cost of a higher latency (higher compute time). For the tracking to work well, the predictions must be fast, so the model can't be too big.
A lower learning rate can help find a better solution, but would require longer training (more epochs).

