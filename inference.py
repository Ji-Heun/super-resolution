import os
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

data_list = ['baby', 'bird', 'butterfly', 'head', 'woman']

for data_name in data_list:
# just edit here ######################################################
upscale_factor = 4
h5_model_path = "/external_home/5_perceptual/train_model5/096_1.49036e+01.h5"
json_filename = '06_inference.json'
json_filepath = "/external_home/5_perceptual/inference/" + json_filename

######################################################################

img_data_path='/external_home/5_perceptual/inference/set5'
data = os.path.join('/external_home/5_perceptual/inference/_set5/' + data_name + '.png')

img = skimage.io.imread(data)
img_hr = img.astype(np.float32)
img_lr = skimage.transform.resize(img_hr,
                                  [img_hr.shape[0] // upscale_factor, img_hr.shape[1] // upscale_factor, img_hr.shape[2]],
                                  order=3)
skimage.io.imsave('2_' + data_name + '_lr.png', img_lr.astype(np.uint8))

bicubic = skimage.transform.resize(img_lr, img_hr.shape, order=3) # order 3 == bi-cubic interpolation #
skimage.io.imsave('2_' + data_name + '_bi.png', bicubic.astype(np.uint8))


# load model and make json file----------------------------------------------------

###########################################################
lossmodel = keras.applications.vgg16.vgg16(include_top=false, # do not include 3fc layers at the end of the network
                                           weights='imagenet', # pre-training on imagenet
                                           input_shape=(288, 288, 3)) # pre-training on imagenet input_shape=(288, 288, 3))

lossmodel.trainable = False
for layer in lossmodel.layers:
    layer.trainable = False

lossmodel = keras.model(lossmodel.inputs, lossmodel.layers[5].output)

def inner_loss_func(y_true, y_pred):
    scaled_y_true = keras.applications.imagenet_utils.preprocess_input(y_true, mode='tf')
    scaled_y_pred = keras.applications.imagenet_utils.preprocess_input(y_pred, mode='tf')
    feature_true = lossmodel(scaled_y_true)
    feature_pred = lossmodel(scaled_y_pred)
    return tf.reduce_mean(tf.square(feature_true - feature_pred))
###########################################################

model1 = keras.models.load_model(h5_model_path, custom_objects={'inner_loss_func': inner_loss_func})
open(json_filename, 'wt').write(model1.to_json())


# edit json file ----------------------------------------------------
with open(json_filepath, 'r') as f:
    json_data = json.load(f)
# print(json.dumps(json_data, indent = "\t"))
# type(json_data) #dic
# type(json_data['config']['layers']) #list
# print(type(json_data['config']['layers'][0]))
# type(json_data['config']['layers'][0])
# print(json_data['config']['layers'][0]['config']['batch_input_shape'])

json_data['config']['layers'][0]['config']['batch_input_shape'] = [None, img_lr.shape[0], img_lr.shape[1], img_lr.shape[2]]
# print(json_data['config']['layers'][0]) check if input shape changed or not

# save edited json file----------------------------------------------------
with open(json_filepath, 'w', encoding='utf-8') as make_file:
    json.dump(json_data, make_file)

# save edited model----------------------------------------------------
model = tf.keras.models.model_from_json(open(json_filepath, 'r').read())
model.load_weights(h5_model_path)

# ---------------------------------------------------- load model
print('img size: ', img_lr.shape)
y_pred = model.predict(img_lr[np.newaxis, ...])
print(np.amax(y_pred)) print(np.amin(y_pred))
#y_pred = np.clip(y_pred, 0, 1)
# print(y_pred.shape) # (1, 256, 256, 3)
# print(y_pred[0,:,:,:].shape) # (256, 256, 3)

img_pred = y_pred[0, :, :, :]
# print(img_pred.shape)
# img_pred *= 255
img_pred = np.clip(img_pred, 0, 255)
# import skimage.exposure as skie
# img_pred = skie.equalize_adapthist(img)
skimage.io.imsave('./model6/' + data_name + '.png', img_pred.astype(np.uint8))