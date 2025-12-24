import os
import random
import cv2
import numpy as np
import tqdm
import glob
from skimage.filters import sobel
from scipy.stats import entropy
import traceback
import sys

root = './datasets/Cholec80/videos'
crop_image = './datasets/Cholec80/picture'

def crop_left(img, crop_point=640):
    # Get the width and height of an image
    height, width = img.shape[:2]

    crop_point = min(crop_point, width)

    # Trim the left portion according to the specified points.
    cropped_img = img[:, crop_point:]

    return cropped_img


def change_size(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2,
                                   19)  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom

    pre1_picture = image[left:left + width, bottom:bottom + height]

    # print(pre1_picture.shape)

    return pre1_picture


def is_image_all_black(img):
    # If the image is empty, it indicates a read failure.
    if img is None:
        print("无法读取图像")
        return False

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check whether the maximum pixel value in the grayscale image is 0 (black).
    max_pixel_value = gray_img.max()

    # If the maximum pixel value is 0, the image is completely black.
    if max_pixel_value == 0:
        return True
    else:
        return False


# 取第一帧
def get_one_image(vid_path, save_path):
    dim = (250, 250)
    count = 0
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(total_frames, fps)
    for i in range(0, total_frames + 1, fps):
        name = int(i / fps) + 1
        img_save_path = os.path.join(save_path, str(name) + '.jpg')
        if os.path.exists(img_save_path):
            #print('{} is exist'.format(img_save_path))
            pass
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if frame is None:
                break
            else:
                frame = crop_left(img=frame)
                frame = change_size(frame)
                frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(img_save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                count += 1

    print('num one frames extracted:', count)


def calculate_laplacian_var(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(grey_img, cv2.CV_64F).var()
    return var


def calculate_entropy_coloured(img):
    ### averages entropies for each colour channel
    channel_entropies = [entropy(np.histogram(img[:, :, i], bins=256, range=(0, 256))[0]) for i in range(img.shape[2])]
    average_entropy = np.mean(channel_entropies)
    return average_entropy


def calculate_sobel_edge_coloured(img):
    sobel_edges = [sobel(img[:, :, i]) for i in range(img.shape[2])]
    return np.mean(np.mean(sobel_edges, axis=0))


# Use Sobel + Vol to select the highest frame
def get_filtered(vid_path, save_path):
    dim = (250, 250)
    count_one = 0
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(total_frames, fps)
    for i in range(fps, total_frames + 1, fps):
        vals = {}
        name = int(i / fps)
        img_save_path = os.path.join(save_path, str(name) + '.jpg')
        if os.path.exists(img_save_path):
            print('{} is exist'.format(img_save_path))
            pass
        else:
            for j in range(i - fps, i):
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                _, frame = cap.read()
                if frame is None:
                    break
                else:
                    frame = crop_left(img=frame)
                    frame = change_size(frame)
                    if frame.shape[0] == 0 or frame.shape[1] == 0:
                        break
                    else:
                        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
                        val = round(calculate_laplacian_var(frame) + calculate_sobel_edge_coloured(frame), 3)
                        vals[val] = j
            if len(vals) == 0:
                break
            else:
                tmp = list(vals.keys())
                tmp.sort(reverse=True)
                print(vals[tmp[0]])
                cap.set(cv2.CAP_PROP_POS_FRAMES, vals[tmp[0]])
                _, frame1 = cap.read()
                frame1 = crop_left(img=frame1)
                frame1 = change_size(frame1)
                frame1 = cv2.resize(frame1, dim, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(img_save_path, frame1, [cv2.IMWRITE_JPEG_QUALITY, 100])
                count_one += 1

    print('num one frames extracted:', count_one)

if __name__ == '__main__':

    cats = os.listdir(root)

    for idx, cat in enumerate(tqdm.tqdm(cats)):
        cat_path = os.listdir(os.path.join(root, cat))
        for name in cat_path:
            print(name)
            mp4_input = os.path.join(root, cat, name)
            output_path = os.path.join(crop_image, name[:-4])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            get_filtered(mp4_input, output_path)
            #get_one_image(mp4_input, output_path)
