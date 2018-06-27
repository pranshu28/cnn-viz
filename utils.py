import numpy as np
import cv2
from keras import backend as K


# Normalize the input and clip between (0,1)
def normalize_clip(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    return x


# Normalize and Convert to RGB image
def deprocess_image(x):
    x = normalize_clip(x)
    x *= 255
    if x.shape[2] != 3:
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# L2 normalization of gradients
def l2_normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


# Gaussian Blur Regularization
def blur_regularization(img, size=(3, 3)):
    return cv2.blur(img, size)


# L2 decay regularization
def decay_regularization(img, decay=0.8):
    return decay * img


# Clipping pixels with small norm
def clip_weak_pixel_regularization(img, percentile=1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped


# Save list of images in one figure
def all_imgs_in_one(img_list, img_width, img_height):
    margin = 5
    n = int(len(img_list) ** 0.5)
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    all_in_one = np.zeros((width, height, 3))

    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(img_list):
                img = img_list[i * n + j]
                all_in_one[(img_width + margin) * i: (img_width + margin) * i + img_width,
                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
    return all_in_one


def save_occs(occ_img_list, img_width, img_height, img_path):
    occ_img = all_imgs_in_one(occ_img_list, img_width, img_height)
    cv2.imwrite('occ_exp/occs_' + img_path, occ_img)


def save_filters(filters, img_width, img_height, layer, name):
    stitched_filters = all_imgs_in_one(filters, img_width, img_height)
    cv2.imwrite('cnn_filters/filters_pro_' + layer + '_' + name, stitched_filters)
