## CNN Visualization (Keras) ##

### Filter visualization - using gradient ascent algorithm
  
  This method generates a synthetic image that maximally activates a neuron.


    usage: viz_gradient_ascent.py [--iterations ITERATIONS] [--img IMG] [--weights_path WEIGHTS_PATH] [--layer LAYER] [--num_filters NUM_FILTERS] [--size SIZE]

    Arguments:
      --iterations INT
      Number of gradient ascent iterations
      --img STRING
      Path to image. If not specified, uses a random init
      --weights_path STRING
      Path of the saved pre-trained model
      --layer STRING
      Name of layer to use
      --num_filters INT
      Number of filters to vizualize
      --size INT
      Image size


### Sensitivity of model - Occlusion experiment 
  
  It finds the part of as input image that a neuron (usually class neuron in output layer) responds to.

    usage: viz_occlusion.py [--img IMG] [--weights_path WEIGHTS_PATH] [--size SIZE] [--occ_size OCC_SIZE] [--pixel PIXEL] [--stride STRIDE] [--norm NORM] [--percentile PERCENTILE]

    Arguments:
      --img STRING
      Path of the input image      
      --weights_path STRING   
      Path of the saved pre-trained model      
      --size INT
      Layer name      
      --occ_size INT      
      Size of occluding window      
      --pixel INT
      Occluding window - pixel values      
      --stride INT   
      Occlusion Stride      
      --norm INT
      Normalize probabilities first      
      --percentile INT
      Regularization percentile for heatmap