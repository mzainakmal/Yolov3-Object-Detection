import tensorflow as tf
import numpy as np
from model import (
Yolov3, YoloLoss,
anchors, anchor_masks
)

model_dir = 'path to model weight that has been converted to the actual model'
learning_rate = 1e-3

def new_custom_model(size, num_classes, transfer='none'):
    # transfer choices = 'none', 'fine_tune', 'no_output'
    yolov3_new = Yolov3(size=size, classes=num_classes)
    anchors_ = anchors
    masks = anchor_masks

    # configure model for transfer learning
    if transfer == 'none':
        pass
    elif transfer == 'fine_tune' and num_classes == 80:
        yolov3_new.load_weights(model_dir)
        darknet = yolov3_new.get_layer('yolo_darknet')
        darknet.trainable = False
    elif transfer == 'no_output' or num_classes != 80:
        base_model = Yolov3(size=size, classes=80)
        base_model.load_weights(model_dir)
        for l in yolov3_new.layers:
            if not l.name.startswith('output'):
                l.set_weights(base_model.get_layer(l.name).get_weights())
                l.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = [YoloLoss(anchors_[mask], classes=num_classes) for mask in masks]

    yolov3_new.compile(optimizer=optimizer,
                       loss=loss)

    return yolov3_new

def train_model(model, train_set, val_set):
    # set callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
    ]
    # train the model
    start_time = time.time()
    epochs = 10
    history = model.fit(train_set,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_set)
    end_time = time.time() - start_time
    print(f'Total Training Time: {end_time}')
