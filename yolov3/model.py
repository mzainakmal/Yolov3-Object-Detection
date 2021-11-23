import tensorflow as tf
import numpy as np

# normalize the anchors for universal use of various image sizes
# anchor sizes are taken from the original yolov3 paper
anchors = np.array([(10, 13), (16, 30), (33, 23),
                    (30, 61), (62, 45), (59, 119),
                    (116, 90), (156, 198), (373, 326)],
                   np.float32) / 416

anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_max_boxes = 120
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# Convolution Block Function
def Conv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=size,
                               strides=strides, padding=padding,
                               use_bias=not batch_norm, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


# Residual Block Function
def Residual(x, filters):
    x1 = Conv(x, filters//2, 1)
    x1 = Conv(x1, filters, 3)
    return tf.keras.layers.Add()([x, x1])


# One Complete Darknet Residual Block
def DarknetRes(x, filters, blocks):
    x = Conv(x, filters, 3, 2)
    for _ in range(blocks):
        x = Residual(x, filters)
    return x


# Full Main Darknet Model
def Darknet(name=None):
    x = inputs = tf.keras.layers.Input([None, None, 3])
    x = Conv(x, 32, 3)
    x = DarknetRes(x, 64, 1)
    x = DarknetRes(x, 128, 2)
    x = x1 = DarknetRes(x, 256, 8)
    x = x2 = DarknetRes(x, 512, 8)
    x = x3 = DarknetRes(x, 1024, 4)
    return tf.keras.Model(inputs, (x1, x2, x3), name=name)


### ---- Output Processing Layer ---- ###


# First Output Processing Block
def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = tf.keras.layers.Input(x_in[0].shape[1:]), tf.keras.layers.Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = Conv(x, filters, 1)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Concatenate()([x, x_skip])
        else:
            x = inputs = tf.keras.layers.Input(x_in.shape[1:])

        x = Conv(x, filters, 1)
        x = Conv(x, filters*2, 3)
        x = Conv(x, filters, 1)
        x = Conv(x, filters*2, 3)
        x = Conv(x, filters, 1)

        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_conv


# Final Output Processing Block
# Including reshaping output into shape [batch_size, gridx, gridy, anchors, (x, y, w, h, obj, ...classes)]

def YoloOut(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = tf.keras.layers.Input(x_in.shape[1:])
        x = Conv(x, filters*2, 3)
        x = Conv(x, anchors*(5+classes), 1, batch_norm=False) #usually 255 filters since anchors=3, classes=80, thus 3*(80+5)=255
        # reshape layer
        x = tf.keras.layers.Lambda(lambda x:tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, 5+classes)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


# Full Darknet Model including the output processing model

def Yolov3(size=None, channels=3, anchors=anchors, masks=anchor_masks,
           classes=80):
    x = input = tf.keras.layers.Input([size, size, channels], name='input')

    x1, x2, x3 = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='conv_small')(x3)
    x_small = YoloOut(512, len(masks[0]), classes, name='output_small')(x)

    x = YoloConv(256, name='conv_med')((x, x2))
    x_med = YoloOut(256, len(masks[1]), classes, name='output_med')(x)

    x = YoloConv(128, name='conv_big')((x, x1))
    x_big = YoloOut(128, len(masks[2]), classes, name='output_big')(x)

    # small, med, big refers to the grid size of the output
    return tf.keras.Model(input, (x_small, x_med, x_big), name='yolov3')


# Creating model
#yolov3_model = Yolov3()
#yolov3_model.summary()


### ---- Output Processing ---- ###

# meshgrid function to imitate grids on the image
def meshgrid_(x, y):
    x, y = tf.meshgrid(tf.range(x), tf.range(y))
    grid = tf.stack((x, y), axis=-1)
    return grid


# Function to process the model ouputs
def YoloBox(pred, anchors, classes):
    # pred: [batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes)]
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, obj, classprob = tf.split(pred, [2, 2, 1, classes], axis=-1)

    box_xy = tf.sigmoid(box_xy) # in range(0,1)
    obj = tf.sigmoid(obj) # in range (0,1)
    classprob = tf.sigmoid(classprob) #in range (0,1)
    boxpred = tf.concat((box_xy, box_wh), axis=-1) #for loss in training

    '''REMINDER: the anchors have been normalized with 416'''
    grid = meshgrid_(grid_size[1], grid_size[0])
    grid = tf.cast(grid, tf.float32)
    grid = tf.expand_dims(grid, axis=2)

    box_xy = (box_xy+tf.cast(grid, tf.float32))/tf.cast(grid_size, dtype=tf.float32) # in range (0,1)
    box_wh = tf.exp(box_wh)*anchors

    box_x1y1 = box_xy-box_wh/2
    box_x2y2 = box_xy+box_wh/2
    bbox = tf.concat((box_x1y1, box_x2y2), axis=-1)

    return bbox, obj, classprob, boxpred


# Function to apply Non Max Suppression to boxes
def YoloNMS(outputs, anchors, masks, classes):
    b, c, t = [], [], []
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)  # [1, grid*grid*anchors, coordinates(4)]
    conf = tf.concat(c, axis=1)  # [1, grid*grid*anchors, conf(1)]
    classprob = tf.concat(t, axis=1)  # [1, grid*grid*anchors, classprob(80)]

    if classes == 1:
        scores = conf
    else:
        scores = conf * classprob

    classes = tf.argmax(scores, 2)  # [1, grid*grid*anchors, maxprob_class(1)]
    scores = tf.reduce_max(scores, [-1])  # [1, grid*grid*anchors, scores(1)]

    box_yx = bbox[0]

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox[0],  # [grid*grid*anchors, coordinates(4)]
        scores=scores[0],  # [grid*grid*anchors, scores(1)]
        max_output_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold,
        soft_nms_sigma=0.4
    )

    num_valid_boxes = tf.shape(selected_indices)[0]  # integer
    selected_boxes = tf.gather(bbox[0], selected_indices)  # [num_valid_boxes, 4]
    selected_classes = tf.gather(classes[0], selected_indices)  # [num_valid_boxes, 1]

    return selected_boxes, selected_scores, selected_classes, num_valid_boxes


# Full function to process the model output into bounding boxes, scores, and classes
def yolo_box_pred(model, anchors=anchors, masks=anchor_masks, classes=80):
    def detect(img):
        preds = model.predict(img)

        batch_size = img.shape[0]
        predictions = []
        # Find Bounding Box for every image in a batch
        for i in range(batch_size):
            out0 = tf.expand_dims(preds[0][i], axis=0)
            out0 = YoloBox(out0, anchors[masks[0]], classes)
            out1 = tf.expand_dims(preds[1][i], axis=0)
            out1 = YoloBox(out1, anchors[masks[1]], classes)
            out2 = tf.expand_dims(preds[2][i], axis=0)
            out2 = YoloBox(out2, anchors[masks[2]], classes)

            outputs = YoloNMS((out0, out1, out2), anchors, masks, classes)
            predictions.append(outputs)

        return predictions
    return detect


### ---- Loss Function ---- ###

# function to find iou between prediction boxes and true boxes
def broadcast_iou(box1, box2):
    # box1: pred boxes with shape (batch, grid, grid, anchors, 4)
    # box2: true boxes with shape (num of true box with an object (N), 4)
    # expand dimension of both tensors so they can be broadcasted
    box1 = tf.expand_dims(box1, -2)
    box2 = tf.expand_dims(box2, 0)

    # Find the right shape to broadcast the two tensors
    broadcast_shape = tf.broadcast_dynamic_shape(tf.shape(box1), tf.shape(box2))
    box1 = tf.broadcast_to(box1, broadcast_shape)
    box2 = tf.broadcast_to(box2, broadcast_shape)

    # Calculate iou
    int_w = tf.maximum(tf.minimum(box1[..., 2], box2[..., 2]) - tf.maximum(box1[..., 0], box2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box1[..., 3], box2[..., 3]) - tf.maximum(box1[..., 1], box2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box_2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    return int_area/(box_1_area + box_2_area -int_area)


# create yolo loss function using inner function
# so then it can be used for training

def YoloLoss(anchors, classes, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all prediction outputs
        pred_box, pred_obj, pred_class, pred_xywh= YoloBox(y_pred, anchors, classes)
        # split xy and wh
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs to xy and wh
        true_box, true_obj, true_class_idx = tf.split(y_true, [4, 1, 1], axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4])/2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        #give higher loss to small boxes
        box_loss_scale = 2-true_wh[..., 0] * true_wh[..., 1]

        # 3. invert true_box to the same format as model output
        grid_size = tf.shape(y_true)[1]
        grid = meshgrid_(grid_size, grid_size)
        grid = tf.expand_dims(tf.convert_to_tensor(grid, dtype=tf.int32), axis=2)
        true_xy = true_xy * tf.cast(grid_size, dtype=tf.float32) - tf.cast(grid, tf.float32)
        #print(true_wh.shape)
        true_wh = tf.math.log(true_wh / tf.convert_to_tensor(anchors, tf.float32))
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        # true_obj in shape [batch, grid, grid, anchors, obj(1)] -> [batch, grid, grid, anchors]
        obj_mask = tf.squeeze(true_obj, axis=-1)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32
            )
        ignore_mask = tf.cast(best_iou<ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = obj_mask * tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        noobj_loss = (1 - obj_mask) * tf.keras.losses.binary_crossentropy(true_obj, pred_obj) * ignore_mask
        class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_class_idx, pred_class)

        # 6. sum all losses
        total_loss = tf.reduce_sum(xy_loss) + tf.reduce_sum(wh_loss) + tf.reduce_sum(obj_loss) + tf.reduce_sum(noobj_loss) + tf.reduce_sum(class_loss)
        return total_loss
    return yolo_loss

