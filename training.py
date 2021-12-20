import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "./tfrecords"
BATCH_SIZE = 128
IMAGE_SIZE = [480, 640]

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train4.tfrecords")
VALID_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/valid4.tfrecords")

'''
for s in VALID_FILENAMES: 
    try: 
        records_n = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(s))
        print(s,': ', records_n) 
    except Exception: 
        print(s,'corrupted') '''


def decode_image(image, height, width, depth):
    image = tf.image.decode_jpeg(image)
    # print(height, width, depth)
    image = tf.cast(image, tf.float32)
    print(depth)
    if depth == 1:
        image = tf.image.grayscale_to_rgb(image)  # print(image) # print(image.shape[2])
        image = tf.reshape(image, [height, width, 3])
        return image


def read_tfrecord(example):
    tfrecord_format = (
        {
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "image_raw": tf.io.FixedLenFeature([], tf.string),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "depth": tf.io.FixedLenFeature([], tf.int64),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    height, width, depth = example['height'], example['width'], example['depth']
    example["image_raw"] = decode_image(example["image_raw"], height, width, depth)

    y_image = tf.image.resize(example["image_raw"], size=(288, 288))
    x_image = tf.image.resize(y_image,
                              [y_image.shape[0] // 4, y_image.shape[1] // 4, ],
                              method='bicubic')
    return x_image, y_image


''' 
def prepare_sample(features): 
    y_image = tf.image.resize(features["image_raw"], size=(288, 288)) 
    x_image = tf.image.resize(y_image, [y_image.shape[0] // 4, y_image.shape[1] // 4], method = 'bicubic') 
    print(x_image.shape) print(y_image.shape) 
    return x_image, y_image '''


def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    # dataset = dataset.map(prepare_sample, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(read_tfrecord,num_parallel_calls=AUTOTUNE)

    # print(dataset)
    return dataset


def get_dataset(filenames, is_train=True):
    dataset = load_dataset(filenames)
    if is_train:
        dataset = dataset.shuffle(2048)
        dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    # print(dataset)
    return dataset


train_dataset = get_dataset(TRAINING_FILENAMES)
valid_dataset = get_dataset(VALID_FILENAMES, is_train=False)
##########################################
### Create Image Transform Model upsampling factor4 ###
inputs = layers.Input(shape=(72, 72, 3))
block1 = layers.Conv2D(filters=64, kernel_size=(9, 9), activation='relu', strides=(1, 1), padding='SAME')(inputs)

# Residal block
for i in range(4):
    x = keras.layers.BatchNormalization()(block1)
    x = keras.layers.Activation(activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='SAME')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='SAME')(x)
    block1 = layers.add([x, block1])

x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='SAME')(block1)
x = tf.nn.depth_to_space(x, 2)

x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='SAME')(x)
x = tf.nn.depth_to_space(x, 2)

outputs = layers.Conv2D(filters=3, kernel_size=(9, 9), activation='relu', strides=(1, 1), padding='SAME')(x)

mainModel = keras.models.Model(inputs, outputs)
# mainModel.summary()

### Create Loss Model (VGG16) ###
lossModel = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(288, 288, 3))
lossModel.trainable = False  # Freeze the outer model
for layer in lossModel.layers:  # freeze the layer, not to train
    layer.trainable = False

# print(lossModel.layers[5])
# relu2_2 from the VGG-16
lossModel = keras.Model(lossModel.inputs, lossModel.layers[5].output)

# lossModel.summary()

def outer_loss_func(lossModel):
    def inner_loss_func(y_true, y_pred):
        scaled_y_true = keras.applications.imagenet_utils.preprocess_input(y_true, mode='tf')
        scaled_y_pred = keras.applications.imagenet_utils.preprocess_input(y_pred, mode='tf')
        feature_true = lossModel(scaled_y_true)
        feature_pred = lossModel(scaled_y_pred)
        return tf.reduce_mean(tf.square(feature_true - feature_pred)) + tf.reduce_mean(
            tf.square(scaled_y_true - scaled_y_pred) / 30)

    return inner_loss_func


### Compile Full Model ###
optimizer = keras.optimizers.Adam(learning_rate=0.001)
mainModel.compile(loss=outer_loss_func(lossModel), optimizer=optimizer)
# mainModel.summary()
# non-trainable params: 260,160, relu2_2

##########################################
print('#####' * 10)
net_input, net_output = next(iter(train_dataset))
print(net_input.shape)
print(net_output.shape)
print('#####' * 10)
print(mainmodel.input.shape)
print(mainmodel.output.shape)

# import shutil
# shutil.rmtree(r"/external_home/5_perceptual/train_5/")

model_dir = './train_model5'
checkpoint_cb = tf.keras.callbacks.modelcheckpoint(filepath=model_dir + "/{epoch:03d}_{loss:.5e}.h5",
                                                   monitor='val_loss', save_best_only=True)
# early_stopping_cb = tf.keras.callbacks.earlystopping(patience=3, restore_best_weights=true)
tensorboard = tf.compat.v1.keras.callbacks.tensorboard(log_dir=model_dir + '/board', profile_batch=2)

mainmodel.fit(train_dataset,
              steps_per_epoch=81920 // BATCH_SIZE,
              # batch_size=512,
              epochs=100,
              validation_data=valid_dataset,
              validation_steps=20480 // BATCH_SIZE,
              callbacks=[checkpoint_cb, tensorboard],  # early_stopping_cb
              )
