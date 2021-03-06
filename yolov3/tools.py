import tensorflow as tf
import cv2
import numpy as np


# Function to preprocess input image(s)
def preprocess_imgs(imgs, size):
    imgs = tf.image.resize(imgs, (size, size))
    imgs = imgs/255
    return imgs


# Function to draw bounding box for a single image
def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    #boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    boxes = tf.clip_by_value(boxes, 0., 1.)
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


