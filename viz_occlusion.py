import argparse
import math
import time

import matplotlib.pylab as plt
import seaborn as sns

from model import *
from utils import *


def get_occ_imgs(img, img_size, occ_size, occ_pixel, occ_stride, classes):
    # Get original image
    image = cv2.imread(img)
    image = cv2.resize(image, (img_size, img_size)).astype(np.float32)

    # Index of class with highest probability
    class_index = np.argmax(classes)
    print('True class index:', class_index)

    # Define number of occlusions in both dimensions
    output_height = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    output_width = int(math.ceil((img_size - occ_size) / occ_stride + 1))
    print('Total iterations:', output_height, '*', output_width, '=', output_height * output_width)

    # Initialize probability heatmap and occluded images
    temp_img_list = []
    prob_matrix = np.zeros((output_height, output_width))

    start = time.time()

    for h in range(output_height):
        for w in range(output_width):
            # Occluder window:
            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(img_size, h_start + occ_size)
            w_end = min(img_size, w_start + occ_size)

            # Getting the image copy, applying the occluding window and classifying it:
            occ_image = image.copy()
            occ_image[h_start:h_end, w_start:w_end, :] = occ_pixel
            predictions = pred_prob_list(model, occ_image.copy())[0]
            prob = predictions[class_index]

            # Collect the probability value in a matrix
            prob_matrix[h, w] = prob

            # Collect occluded images   
            occ_image[h_start:h_end, w_start:w_end, :] = prob*255
            cv2.putText(img=occ_image, text=str(round(prob,4)), org=(w_start, int(h_start + (h_end - h_start) / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255*(1-prob),255*(1-prob),255*(1-prob)), thickness=1)  
            cv2.imwrite('occ_exp/video/'+img_name+str(h*output_width+w+1).zfill(6)+'.png',occ_image) 
            
            # To save occluded images as a video, run the following shell command
            """ffmpeg -framerate 5 -i occ_exp/video/<img_name>%06d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p occ_exp/<img_name>.mp4"""

            temp_img_list.append(occ_image)

        print('Percentage done :', round(((h + 1) * output_width) * 100 / (output_height * output_width), 2), '%')

    end = time.time()
    elapsed = end - start
    print('Total time taken:', elapsed, 'sec\tAverage:', elapsed / (output_height * output_width), 'sec')

    # Save probabilities and all occluded images in one
    np.save('occ_exp/probs_' + img_name + '.npy', prob_matrix)
    # save_occs(temp_img_list, img_size, img_size, img_path.split('/')[-1])

    return prob_matrix


def regularize(prob, norm, percentile):
    # First save the original prob matrix as heat-map
    f = plt.figure(1)
    sns.heatmap(prob, xticklabels=False, yticklabels=False)
    f.savefig('occ_exp/heatmap_' + img_path.split('/')[-1])

    # Apply Regularization
    prob = normalize_clip(prob) if norm else prob
    clipped = clip_weak_pixel_regularization(prob, percentile=percentile)
    reg_heat = blur_regularization(1 - clipped, size=(3, 3))

    # Save regularized heat-map
    f2 = plt.figure(2)
    sns.heatmap(reg_heat, xticklabels=False, yticklabels=False)
    f2.savefig('occ_exp/heatmap_reg_' + img_path.split('/')[-1])

    return reg_heat


def join(heat_reg, img, img_size, occ_size):
    # Get original image
    image = cv2.imread(img, 1)
    inp_img = cv2.resize(image, (img_size, img_size))

    H, W = image.shape[0], image.shape[1]
    bord = int(occ_size / 2)

    # Define heat-map to be projected on original image
    heat_map = cv2.resize(heat_reg,(img_size, img_size)).astype(np.float32)
    
    # Second way to define heat-map - manually set border values
    # heat_map = np.zeros((img_size, img_size))
    # heat_map[bord:img_size - bord, bord:img_size - bord] = cv2.resize(heat_reg,
    #     (img_size - occ_size, img_size - occ_size)).astype(np.float32)
    # np.place(heat_map, heat_map == 0.0, np.median(heat_map))

    # Third way to define heat-map - replicate border values
    # heatmap = cv2.resize(heat, (img_size-occ_size, img_size-occ_size)).astype(np.float32)
    # heatmap = cv2.copyMakeBorder(heat-map,bord,bord,bord,bord,cv2.BORDER_REPLICATE)


    # Original image * heat-map
    for i in range(3):
        inp_img[:, :, i] = heat_map * inp_img[:, :, i]
    inp_viz = cv2.resize(inp_img, (W, H))

    # Save the final output
    cv2.imwrite('occ_exp/final_' + img.split('/')[-1], inp_viz)

    return inp_viz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str)  # Path of the input image
    parser.add_argument('--weights_path', type=str, default='vgg16_weights.h5')  # Path of the saved pre-trained model
    parser.add_argument('--size', type=int, default='224')  # Layer name
    parser.add_argument('--occ_size', type=int, default='40')  # Size of occluding window
    parser.add_argument('--pixel', type=int, default='0')  # Occluding window - pixel values
    parser.add_argument('--stride', type=int, default='5')  # Occlusion Stride
    parser.add_argument('--norm', type=int, default='1')  # Normalize probabilities first
    parser.add_argument('--percentile', type=int, default='25')  # Regularization percentile for heatmap
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print('\n', args)

    img_path, img_size = args.img, args.size
    img_name = img_path.split('/')[-1].split('.')[0]
    occ_size, occ_pixel, occ_stride = args.occ_size, args.pixel, args.stride

    # Input pre-trained model, defined in model.py
    model = load_trained_model(args.weights_path)

    # Get original image
    input_image = cv2.imread(img_path)
    input_image = cv2.resize(input_image, (img_size, img_size)).astype(np.float32)

    # Get probability list and print top 5 classes
    result = pred_prob_list(model, input_image)
    de_result = decode_predictions(result)[0]
    print('\nPredicted: ', de_result)

    # Start occlusion experiment and store predicted probabilities in a file
    print('Running occlusion iterations (Class:', de_result[0][1], ') ...\n')
    probs = get_occ_imgs(img_path, img_size, occ_size, occ_pixel, occ_stride, result)

    # Get probabilities and apply regularization
    print('\nGetting probability heat-map and regularizing...')
    probs = np.load('occ_exp/probs_' + img_name + '.npy')
    heat = regularize(probs, args.norm, args.percentile)

    # Project heatmap on original image
    print('\nProject the heat-map to original image...')
    aug = join(heat, img_path, img_size, occ_size)

    print('\nDone')
