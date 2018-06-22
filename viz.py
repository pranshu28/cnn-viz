import numpy as np
import cv2
import random
from keras import backend as K
from utils import *
from model import *
import argparse

 
# Gaussian Blur Regularization
def blur_regularization(img, grads, size=(3, 3)):
    return cv2.blur(img, size)

# L2 decay regularization
def decay_regularization(img, grads, decay=0.8):
    return decay * img

# Clipping pixels with small norm
def clip_weak_pixel_regularization(img, grads, percentile=1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped


def gradient_ascent_iteration(loss_function, img):

    # Update image with the gradient
    loss_value, grads_value = loss_function([img])
    gradient_ascent_step = img + grads_value * 0.9

    grads_row_major = np.transpose(grads_value[0, :], (1, 2, 0))
    img_row_major = np.transpose(gradient_ascent_step[0, :], (1, 2, 0))

    # Define weights for individual regularization
    regularizations = [blur_regularization, decay_regularization, clip_weak_pixel_regularization]
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)
    
    # Apply regularization on the gradient ascent output image
    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    act = np.sum(weighted_images, axis=0)

    # Difference has been taken to visualize activated part clearly in the filter 
    act = np.float32([np.transpose(act, (2, 0, 1))]) - img

    return act


def visualize_filter(input_img, filter_index, img_placeholder, number_of_iterations=20):
    
    # a loss function to maximize the activation of the filter
    loss = K.mean(layer[:, :, :, filter_index])
    
    # compute the gradient of the input picture wrt loss and normalize it
    grads = K.gradients(loss, img_placeholder)[0]
    grads = normalize(grads) # (utils.py)

    # function to return loss and gradient for given image
    iterate = K.function([img_placeholder], [loss, grads])

    img = input_img * 1
    for i in range(number_of_iterations):
        img = gradient_ascent_iteration(iterate, img)
    
    # function to convert it into a valid image (utils.py)
    img = deprocess_image(img[0])
    print("Done with filter", filter_index)
    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20) # Gradient Ascent Process
    parser.add_argument("--img", type=str) # Path of the input image
    parser.add_argument("--weights_path", type=str, default='vgg16_weights.h5') # Path of the saved pre-trained model
    parser.add_argument("--layer", type=str, default='conv1_1') # Layer name
    parser.add_argument("--num_filters", type=int, default=64) # Number of filters to visualize from specified layer
    parser.add_argument("--size", type=int, default=128) # Size of the image (width, height)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    print(args)
    img_width, img_height = args.size, args.size

    # define input placeholder and get model and weights
    input_placeholder = K.placeholder((1, img_width, img_height, 3))
    first_layer = ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3))
    model = get_model(first_layer)
    input_placeholder = model.input
    model = load_model_weights(model, args.weights_path)

    # Outupt of the specified layer
    layer = get_output_layer(model, args.layer)

    # Initialize input image and resize it
    if args.img is None:
        init_img = np.random.random((1, img_width, img_height, 3)) * 20 + 128.
        cv2.imwrite('random.png', cv2.resize(init_img[0], (img_width * 2, img_height * 2)))
    else:
        img = cv2.imread(args.img, 1)
        img = cv2.resize(img, (img_width, img_height))
        init_img = [img]  # [np.transpose(img, (2, 0, 1))]

    # Randomly choose filters from CNN layer
    filter_indexes = [random.randint(0, layer.shape[3] - 1) for i in range(0, args.num_filters)]
    filter_indexes.sort()

    # To choose filters in sequence
    # filter_indexes = range(0, args.num_filters)
    
    # Iteration for all filters
    vizualizations = [None] * len(filter_indexes)
    for i, index in enumerate(filter_indexes):
        vizualizations[i] = visualize_filter(init_img, index, input_placeholder, args.iterations)
        save_filters(vizualizations, img_width, img_height, args.layer, args.img)
