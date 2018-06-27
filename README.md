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

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/test_image/index.jpeg "Test Image")

Layer Conv1_1 (All 64 filters)

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/cnn_filters/filters_conv1_1_index.jpeg "Layer Conv1_1")

Layer Conv1_2 (All 64 filters)

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/cnn_filters/filters_conv1_2_index.jpeg "Layer Conv1_2")

Layer Conv5_3 (Randomly chosen 16 filters)

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/cnn_filters/filters_conv5_3_index.jpeg "Layer Conv5_3")

#### Sensitivity of model - Occlusion experiment 
  
  It finds the part of an input image that a neuron (usually a class neuron) responds to.

    usage: viz_occlusion.py [--img IMG] [--weights_path WEIGHTS_PATH] [--size SIZE] [--occ_size OCC_SIZE] [--pixel PIXEL] [--stride STRIDE] [--norm NORM] [--percentile PERCENTILE]

    Arguments:
      --img STRING - Path of the input image      
      --weights_path STRING - Path of the saved pre-trained model      
      --size INT - Image size      
      --occ_size INT - Size of occluding window      
      --pixel INT - Occluding window - pixel values      
      --stride INT - Occlusion Stride      
      --norm INT - Normalize probabilities first      
      --percentile INT - Regularization percentile for heatmap

###### Example:

For the given Test Image:

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/test_image/bus.jpg "Test Image")

The pre-trained CNN model predicted `Class: School Bus` with the highest probability of `0.86515594`.    

Probability heat-map for `School bus` class after occlusion experiment:

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/occ_exp/heatmap_bus.jpg "Probability heatmap after occlusion experiment")

Regularized Heat-map

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/occ_exp/heatmap_reg_bus.jpg "Regularized Heatmap")

Projection of heat-map on given image

![alt text](/home/pranshu/Documents/Projects/IITM/cnn-visualization/occ_exp/final_bus.jpg "Projection of Heatmap on given image")
