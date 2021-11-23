import tensorflow as tf
import numpy as np
import cv2
from yolov3.tools import preprocess_imgs, draw_outputs
from yolov3.model import yolo_box_pred

img_path = 'path to image file'

def detections(model, labels, img_path):
    img_raw = tf.image.decode_image(
        open(img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = preprocess_imgs(img, 416)
    trial = yolo_box_pred(model)(img)
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    #img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    img = draw_outputs(img, trial[0], labels)
    print(int(trial[0][-1]))
    for i in range(trial[0][-1]):
      print(labels[int(trial[0][2][i])],
            np.array(trial[0][1][i], np.float32),
            np.array(trial[0][0][i], np.float32))
    cv2.imshow(img, 'detections')
    cv2.waitKey(0)
