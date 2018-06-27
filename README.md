### CNN Visualization (Keras)

#### Filter visualization - using gradient ascent algorithm
  
  This method generates a synthetic image that maximally activates a neuron.


    usage: viz_gradient_ascent.py [--iterations ITERATIONS] [--img IMG] [--weights_path WEIGHTS_PATH] [--layer LAYER] [--num_filters NUM_FILTERS] [--size SIZE]

    Arguments:
      --iterations INT - Number of gradient ascent iterations
      --img STRING - Path of the input image
      --weights_path STRING - Path of the saved pre-trained model
      --layer STRING - Name of layer to use
      --num_filters INT - Number of filters to vizualize
      --size INT - Image size

###### Example:

Test Image:

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/test_image/index.jpeg "Test Image")

Layer Conv1_1 (All 64 filters)

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/cnn_filters/filters_conv1_1_index.jpeg "Layer Conv1_1")

Layer Conv1_2 (All 64 filters)

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/cnn_filters/filters_conv1_2_index.jpeg "Layer Conv1_2")

Layer Conv5_3 (Randomly chosen 16 filters)

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/cnn_filters/filters_conv5_3_index.jpeg "Layer Conv5_3")

#### Sensitivity of model - Occlusion experiment 
  
  It finds the part of an input image that an output neuron responds to. It iterates a black square that occludes various parts of the image and monitors the output of the classifier model. This represents localization of the objects withing the image the probability of the correct class drops as a significant portion of the object is occluded.

    usage: viz_occlusion.py [--img IMG] [--weights_path WEIGHTS_PATH] [--size SIZE] [--occ_size OCC_SIZE] [--pixel PIXEL] [--stride STRIDE] [--norm NORM] [--percentile PERCENTILE]

    Arguments:
      --img STRING - Path of the input image      
      --weights_path STRING - Path of the saved pre-trained model      
      --size INT - Image size      
      --occ_size INT - Size of occluding window      
      --pixel INT - Occluding window - pixel values      
      --stride INT - Occlusion Stride      
      --norm INT - Normalize probabilities before regularization 
      --percentile INT - Regularization percentile for heatmap

###### Example:

For the given Test Image:

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/test_image/bus.jpg "Test Image")

The pre-trained CNN model predicted `Class: School Bus` with the highest probability of `0.86515594`.    

The following figure shows visualization of probabilities output for `School bus` class as a function of occluder position:

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/occ_exp/heatmap_bus.jpg "Probability heatmap after occlusion experiment")

To clearly localize the object, we regularize the above heatmap to extract its strongest features.

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/occ_exp/heatmap_reg_bus.jpg "Regularized Heatmap")

The below image shows projection of regularized heat-map on the input image. It proves that the above visualization genuinely corresponds to the object structure that stimulates these features.

![alt text](https://github.com/pranshu28/cnn-viz/blob/master/occ_exp/final_bus.jpg "Projection of Heatmap on given image")
