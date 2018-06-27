### CNN Visualization (Keras)

#### Filter visualization - using gradient ascent algorithm
  
  This method generates a synthetic image that maximally activates a neuron. We use a test image to visualize what part of the filters in a given convolution layer gets activated during the forward pass. Then, we backpropagate to compute gradients of the neuron values in the filters with respect to image pixels and update the image with these gradients. For better interpretation, we penalize these gradients with L2 norm and apply some more regularization techniques. Now, we subtract the original input image from updated one to visualize the activated part of the filters.


    usage: 
      viz_gradient_ascent.py [--iterations ITERATIONS] [--img IMG] [--weights_path WEIGHTS_PATH] [--layer LAYER] [--num_filters NUM_FILTERS] [--size SIZE]

    Arguments:
      --iterations INT - Number of gradient ascent iterations
      --img STRING - Path of the input image
      --weights_path STRING - Path of the saved pre-trained model
      --layer STRING - Name of layer to use
      --num_filters INT - Number of filters to vizualize
      --size INT - Image size

##### Example:

Suppose the test image is of a bird:

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/test_image/index.jpeg "Test Image")

After one forward pass, if we visualize the filters in first layer (Conv1_1), we can clearly see the bird-like shape in some filters. These shapes corresponds to activated neurons in the filters that further helps the CNN model to recognize objects in the image. 

Layer Conv1_1 (All 64 filters)

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/cnn_filters/filters_conv1_1_index.jpeg "Layer Conv1_1")

Similarly, we visualize the filters in the second layer (Conv1_2) in which the activation maps are noisy but there are still a few filters has bird-like shape of activation map.

Layer Conv1_2 (All 64 filters)

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/cnn_filters/filters_conv1_2_index.jpeg "Layer Conv1_2")

Finally, when we visualize the last convolution layer as it actually determines the output of the model. We select 16 random filters (from 512) out of which there is a filter that convincingly recognizes the shape of the bird with some more details in it.

Layer Conv5_3 (Randomly chosen 16 filters)

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/cnn_filters/filters_conv5_3_index.jpeg "Layer Conv5_3")

#### Sensitivity of model - Occlusion experiment 
  
  It finds the part of an input image that an output neuron responds to. It iterates a blank window that occludes various parts of the image and monitors the output of the classifier model. This representation helps us to localize the objects withing the image due to the fact that when a significant portion of the object is occluded, the probability of the correct class drops.

    usage: 
      viz_occlusion.py [--img IMG] [--weights_path WEIGHTS_PATH] [--size SIZE] [--occ_size OCC_SIZE] [--pixel PIXEL] [--stride STRIDE] [--norm NORM] [--percentile PERCENTILE]

    Arguments:
      --img STRING - Path of the input image      
      --weights_path STRING - Path of the saved pre-trained model      
      --size INT - Image size      
      --occ_size INT - Size of occluding window      
      --pixel INT - Occluding window - pixel values      
      --stride INT - Occlusion Stride      
      --norm INT - Normalize probabilities first 
      --percentile INT - Regularization percentile for heatmap 

##### Example:

For the given test image:

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/test_image/bus.jpg "Test Image")

The pre-trained CNN model predicted `Class: School Bus` with the highest probability of `0.86515594`.    

The following figure shows visualization of probabilities output for `School bus` class as a function of occluder position:

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/occ_exp/heatmap_bus.jpg "Probability heatmap after occlusion experiment")

To clearly localize the object, we regularize the above heatmap to extract the strongest features.

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/occ_exp/heatmap_reg_bus.jpg "Regularized Heatmap")

The below images show before and after projection of regularized heat-map on the input image. It proves that the above visualization genuinely corresponds to the object structure that stimulates these features.

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/test_image/bus.jpg "Test Image")
![alt text](https://github.com/pranshu28/cnn-viz/blob/master/occ_exp/final_bus.jpg "Projection of Heatmap on given image")


#### References:
* Visualizing what ConvNets learn - Andrej Karpathy. ([link](http://cs231n.github.io/understanding-cnn/))
* Convolutional Neural Networks for Visual Recognition - CS231n. ([lecture](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf))
* How convolutional neural networks see the world - Francois Chollet. ([link](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html), [github](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py))
* Visualizing and Understanding Convolutional Networks - Matthew D Zeiler and Rob Fergus. ([paper](https://arxiv.org/abs/1311.2901))
* Occlusion experiments - DaoYu Lin. ([github](https://github.com/BUPTLdy/occlusion_experiments/blob/master/Occlusion_experiments.ipynb))