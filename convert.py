import tensorflow as tf
import numpy as np
from absl import logging
from yolov3.model import Yolov3

# Function to import pretrained model weight from YOLO website

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'conv_small',
    'output_small',
    'conv_med',
    'output_med',
    'conv_big',
    'output_big',
]


def load_darknet_weights(model, weights_file, layers=YOLOV3_LAYER_LIST):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

# Loading weights and testing
yolov3_model = Yolov3()
weight_path = 'path to pretrained weight file'

load_darknet_weights(yolov3_model, weight_path)
print('weights loaded')

img = np.random.random((1,320,320,3)).astype(np.float32)
output = yolov3_model(img)
print('sanity check passed')

weight_dir = 'path to save weights'
yolov3_model.save_weights(weight_dir)
