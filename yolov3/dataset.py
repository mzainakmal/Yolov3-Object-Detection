import tensorflow as tf
import xml.etree.ElementTree as et
import os
from tools import preprocess_imgs

# labels for the new dataset
new_labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']

# function to extract data from xml file
def process_xml(dataset_path, labels):
    img_files = []
    y_true = []
    for xml_file in os.listdir(dataset_path):
        tree = et.parse(os.path.join(dataset_path, xml_file))
        root = tree.getroot()
        # take image filename
        img_files.append(root[1].text)

        imgdat = []
        # loop over every object data in an image
        for obj in root.iter('object'):
            coorobj = []
            for i in range(2):
                # take object cooridinates and normalize to coresponding image size
                coorobj.append(float(obj[5][i*2].text)/float(root[2][0].text)) # x / width
                coorobj.append(float(obj[5][i*2+1].text)/float(root[2][1].text)) # y / height

            coorobj.append(labels.index(obj[0].text))
            imgdat.append(coorobj)

        y_true.append(imgdat)

    return y_true, img_files


# function to convert extracted data to tensor format
def list_to_tensor(y_true):
    # inputs: list of targets
    # ouputs: tensor with shape [N, boxes, (coordinates(4), class)]
    maxlen = 0
    for y in y_true:
        if len(y) > maxlen:
            maxlen = len(y)

    for i in range(len(y_true)):
        dif = maxlen - len(y_true[i])
        # add padding so every item in list are in the same shape
        y_true[i] += [[0, 0, 0, 0, 0]] * dif

    y_true_tensor = tf.convert_to_tensor(y_true, tf.float32)
    return y_true_tensor

def transform_to_output_format(y_true, grid_size, anchor_idxs):
    # inputs: [N, boxes, (coordinates(4), class, best_anchor)]
    # outputs: [N, grid, grid, anchors, (coordinates(4), obj, class)]
    N = y_true.shape[0]

    y_true_out = tf.zeros(
      (N, grid_size, grid_size, anchor_idxs.shape[0], 6)
    )

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    return tf.tensor_scatter_nd_update(
      y_true_out, indexes.stack(), updates.stack())

def transform_targets(y_true, anchors, anchor_masks, size):

    # 1. add anchor index to y_true
    grid_size = size // 32

    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1] #[anchors, 2]
    box_wh = y_true[..., 2:4] - y_true[..., 0:2] #[N, boxes, 2]
    box_wh = tf.tile(tf.expand_dims(box_wh, axis=-2), # expand dims into shape [N, boxes, 1, 2]
                   (1, 1, tf.shape(anchors)[0],1)) # tile into shape [N, boxes, anchors, xy(2)]
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    union = anchor_area + box_area - intersection
    iou = intersection / union
    choosen_anchors = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    y_true = tf.concat((y_true, tf.expand_dims(choosen_anchors, axis=-1)), axis=-1) #[N, boxes, (coordinates(4), class, best_anchor)]
    print(y_true.shape)

    # 2. transform y_true from [N, boxes, (coordinates(4), class, best_anchor)]
    # to [N, grid, grid, anchors, (coordinates(4), obj, class)]
    y_outs = []

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_to_output_format(y_true, grid_size, anchor_idxs))
        grid_size *= 2
        print("a grid done")

    return tuple(y_outs)

# function to load images
def load_imgs(img_path, img_files, size):
    imgs = []
    for img_file in img_files:
        img = tf.image.decode_image(
            open(os.path.join(img_path, img_file), 'rb').read(), channels=3
            )
        img = preprocess_imgs(img, size)
        imgs.append(img)
    imgs = tf.convert_to_tensor(imgs, tf.float32)
    return imgs

# create tensorflow dataset
def create_dataset(img_data, y_output, train_size=0.9):
    dataset = tf.data.Dataset.from_tensor_slices((img_data, y_output))
    dataset = dataset.shuffle(buffer_size=256, seed=17)

    traindata = int(img_data.shape[0] * train_size)
    train_set = dataset.take(traindata)
    train_set = train_set.batch(8)
    val_set = dataset.skip(traindata).take(img_data.shape[0] - traindata)
    val_set = val_set.batch(8)
    return train_set, val_set

# processing xml files
#dat_path = '/content/drive/MyDrive/Future Preparations/Computer Vision Training/Object Detection Project/Dataset/Mask detection/annotations'
#y_coord, y_files = process_xml(dat_path, new_labels)
#y_target = list_to_tensor(y_coord)
#print(y_target.shape)
#y_output = transform_targets(y_target, anchors, anchor_masks, 416)

# processing img files
#img_path = '/content/drive/MyDrive/Future Preparations/Computer Vision Training/Object Detection Project/Dataset/Mask detection/images'
#img_data = load_imgs(img_path, y_files, 416)
#print(img_data.shape)

# creating tensorflow dataset
#train_set, val_set = create_dataset(img_data, y_output)