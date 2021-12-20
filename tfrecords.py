import tensorflow as tf
import os
import json

#image_string = open("/external_home/5_perceptual/train2014/coco_train2014_000000000009.jpg" , 'rb').read()
# #print(tf.io.decode_jpeg(image_string).shape) # (480, 640, 3)

annotations_dir = "/external_home/5_perceptual/_annotations/"
annotation_val = os.path.join(annotations_dir, "instances_val2014.json")
annotation_train = os.path.join(annotations_dir, "instances_train2014.json")
path_dir = "/external_home/5_perceptual/"


with open(annotation_train, "r") as f:
    annotations_train = json.load(f)["annotations"]

with open(annotation_val, "r") as f:
    annotations_val = json.load(f)["annotations"]

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.feature(bytes_list=tf.train.byteslist(value=[value]))

def _float_feature(value):
    return tf.train.feature(float_list=tf.train.floatlist(value=[value]))

def _int64_feature(value):
    return tf.train.feature(int64_list=tf.train.int64list(value=[value]))

def image_example(image_string, example):
    image_shape = tf.io.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'image_raw': _bytes_feature(image_string),
        'image_id': _int64_feature(example["image_id"]),
    }
    return tf.train.example(features=tf.train.features(feature=feature)

with tf.io.tfrecordwriter('train4.tfrecords') as writer:
    a = 0
    for sample in annotations_train:
        image_path = f"{path_dir}{'_train2014/coco_train2014_'}{sample['image_id']:012d}.jpg"
        image_string = open(image_path , 'rb').read()
        tf_example = image_example(image_string, sample)
        image_shape = tf.io.decode_jpeg(image_string).shape
        writer.write(tf_example.serializetostring())

        a +=1
        if a == 81920:
            break
        else :
            continue

with tf.io.tfrecordwriter('valid4.tfrecords') as writer:
    a = 0
    for sample in annotations_val:
        image_path = f"{path_dir}{'_val2014/coco_val2014_'}{sample['image_id']:012d}.jpg"
        image_string = open(image_path , 'rb').read()
        tf_example = image_example(image_string, sample)
        image_shape = tf.io.decode_jpeg(image_string).shape
        writer.write(tf_example.serializetostring())

        a +=1
        if a == 20480:
            break
        else :
            continue

# download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('.'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    os.remove(annotation_zip)

# download image train files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    path = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    path = os.path.abspath('.') + image_folder