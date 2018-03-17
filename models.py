from keras.layers import *
from keras.models import Model

def preproc(x): return (x-np.array([123.68,116.779,103.939],dtype=np.float32))[:,:,:,::-1]*0.017

def conv_block(x,nb_filters,k_size=3,padding="same",activation="relu",batchnorm=True):
    for i in range(2):
        x = Conv2D(nb_filters,
                   kernel_size=(k_size, k_size),
                   strides=(1, 1),
                   padding=padding,
                   activation=activation )(x)

    x = Conv2D(nb_filters,
               kernel_size=(k_size, k_size),
               strides=(2, 2),
               padding=padding,
               activation=activation )(x)

    if batchnorm:
        x = BatchNormalization()(x)
    return x

def cnn_md(IMG_SCALE=108):
    ipt = Input(shape=(IMG_SCALE, IMG_SCALE, 3), name="universal_input")
    x = Lambda(preproc, name="color_preprocessing")(ipt)

    # Conv layers
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    # Flatten and use fully connected layers to compress it to length of 48
    x = Flatten(name="flatten_layer")(x)
    x = Dense(48, name="fc2_160", activation="sigmoid")(x)
    md = Model(ipt, x, name="hash_model2")
    return md
