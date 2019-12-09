import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D
from keras.layers import Concatenate, Activation, GlobalAveragePooling3D, Dropout, Dense
from smoothTransformer import smoothTransformer3D

def getModel(moving, reference):

    i = Concatenate(axis=-1)([moving, reference])

    # encoder
    enc1 = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(i)
    enc1 = LeakyReLU()(enc1)
    enc2 = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(enc1)
    enc2 = LeakyReLU()(enc2)
    enc3 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), kernel_initializer='he_normal', padding='same')(enc2)
    enc3 = LeakyReLU()(enc3)
    enc = Concatenate(axis=-1)([i,enc1,enc2,enc3])
    enc = Dropout(0.5)(enc)

    # decoder
    ddec = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), kernel_initializer='he_normal', padding='same')(enc)
    ddec = LeakyReLU()(ddec)
    ddec = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = LeakyReLU()(ddec)
    ddec = Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = LeakyReLU()(ddec)

    # Affine
    affine = GlobalAveragePooling3D()(enc)
    affine = Dense(12, bias_initializer='zeros', kernel_initializer='zeros')(affine)
    affine = Activation('linear')(affine)                   # this should be linear and initialized to 0 !

    # Deformable
    deformable = Conv3D(3, (3, 3, 3), padding='same', bias_initializer='zeros', kernel_initializer='zeros')(ddec)
    deformable = Activation('linear')(deformable)           # this should be linear and initialized to 0 !
    
    [deformed, sampling_grid] = smoothTransformer3D(maxgrad=4)([moving, deformable, affine])
    return deformed, sampling_grid


if __name__ == '__main__':

    # input
    moving = Input((32,32,32,1))
    reference = Input((32,32,32,1))
    deformed, sampling_grid = getModel(moving, reference)
    model = Model(input=[moving, reference], output=deformed)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    get_sampling_grid = K.function([moving, reference], [sampling_grid])

    x = np.random.random((1,32,32,32,1))    # moving
    y = np.random.random((1,32,32,32,1))    # reference
    r = model.predict([x,y])                # reconstructed
    d = get_sampling_grid([x,y])            # sampling_grid

    # With no training the network should apply the identity transform to the moving image
    print(d) # d should be a identity sampling grid
