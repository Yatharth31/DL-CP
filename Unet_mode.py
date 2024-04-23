from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, multiply
from keras.models import Model
import os
import glob
import io
import imageio
import sklearn.model_selection as sk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import PIL
import h5py
import tensorflow as tf
from tensorflow import keras
def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

    f = Activation('relu')(add([theta_x, phi_g]))

    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x, rate])

    return att_x

def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    return x

def encoder(x):
    skips = []
    for i in range(3):
        x = conv_block(x, 64)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
    return x, skips

def decoder(x, skip):
    for i in range(3):
        x = UpSampling2D((2, 2))(x)
        x = concatenate([skip[i], x])
        x = attention_block(x, skip[i], 64)
        x = conv_block(x, 64)
    return x

def build_model():
    inputs = Input((18,344,315,1))
    x, skips = encoder(inputs)
    x = decoder(x, reversed(skips))
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    return Model(inputs, outputs)

model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy')

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# larger possible dpi: 382x350
def create_dataset_from_raw(directory_path, resize_to):
    resize_width = resize_to[0]
    resize_height = resize_to[1]
    batch_names = [directory_path + name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    dataset = np.zeros(shape=(len(batch_names),36,resize_height,resize_width)) # (samples, filters, rows = height, cols = width)

    for batch_idx,batch in enumerate(batch_names):
        files = [x for x in os.listdir(batch) if x != '.DS_Store']
        files.sort()
        crn_batch = np.zeros(shape=(36, resize_height, resize_width))
        for (idx,raster) in enumerate(files):
            fn = batch + '/' + raster
            img = h5py.File(fn)
            original_image = np.array(img["image1"]["image_data"]).astype(float)
            img = Image.fromarray(original_image)
            # note that here it is (width, heigh) while in the tensor is in (rows = height, cols = width)
            img = img.resize(size=(resize_width, resize_height))
            original_image = np.array(img)
            original_image = original_image / 255.0
            crn_batch[idx] = original_image
        dataset[batch_idx] = crn_batch
        print("Importing batch:" + str(batch_idx+1))
    return dataset

def split_data_xy(data):
    x = data[:, 0 : 18, :, :]
    y = data[:, 18 : 36, :, :]
    return x, y

dataset = create_dataset_from_raw('./data/images/raw_training/', resize_to=(315,344))
dataset = np.expand_dims(dataset, axis=-1)
dataset_x, dataset_y = split_data_xy(dataset)
X_train, X_val, y_train, y_val = sk.train_test_split(dataset_x,dataset_y,test_size=0.1, random_state = 42)


EPOCHS = 25
BATCH_SIZE = 1

#Fit the model
model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=1,
)
model.save('./model_saved')