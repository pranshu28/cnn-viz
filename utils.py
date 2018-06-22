import numpy as np
import cv2
from keras import backend as K


def save_filters(filters, img_width, img_height, layer, name):
    name = 'random.png' if name == None else name
    margin = 5
    n = int(len(filters) ** 0.5)
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

     # Insert the filters in image
     for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(filters):
                img = filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # Save the final result
    cv2.imwrite('output/filters_'+ layer + '_' + name, stitched_filters)


def deprocess_image(x):
    # normalize
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    # clip between [0,1]
    x += 0.5
    x = np.clip(x, 0, 1)
    
    # convert to RGB array
    x *= 255
    if x.shape[2] != 3:
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# L2 normalization of gradients
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
