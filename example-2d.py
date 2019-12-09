import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers import Concatenate, Activation, GlobalAveragePooling2D, Dropout, Dense
from smoothTransformer import smoothTransformer2D

import matplotlib.pylab as plt

def getModel(moving, reference):

    i = Concatenate(axis=-1)([moving, reference])

    # encoder
    enc1 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(i)
    enc1 = LeakyReLU()(enc1)
    enc2 = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(enc1)
    enc2 = LeakyReLU()(enc2)
    enc3 = Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(enc2)
    enc3 = LeakyReLU()(enc3)
    enc = Concatenate(axis=-1)([i,enc1,enc2,enc3])
    enc = Dropout(0.5)(enc)

    # decoder
    ddec = Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(enc)
    ddec = LeakyReLU()(ddec)
    ddec = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = LeakyReLU()(ddec)
    ddec = Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = LeakyReLU()(ddec)

    # Affine
    affine = GlobalAveragePooling2D()(enc)
    affine = Dense(9, bias_initializer='zeros', kernel_initializer='zeros')(affine)
    affine = Activation('linear')(affine)                   # this should be linear and initialized to 0 !

    # Deformable
    deformable = Conv2D(2, (3, 3), padding='same', bias_initializer='zeros', kernel_initializer='zeros')(ddec)
    deformable = Activation('linear')(deformable)           # this should be linear and initialized to 0 !
    
    [deformed, sampling_grid] = smoothTransformer2D(maxgrad=4)([moving, deformable, affine])
    return deformed, sampling_grid


if __name__ == '__main__':

    # get model
    moving = Input((28,28,1))
    reference = Input((28,28,1))
    deformed, sampling_grid = getModel(moving, reference)
    model = Model(input=[moving, reference], output=deformed)
    model.compile(optimizer='adam', loss='mse')
    get_sampling_grid = K.function([moving,reference], [sampling_grid])

    # get data
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    y_train = x_train[np.random.permutation(x_train.shape[0]),...]
    y_test = x_test[np.random.permutation(x_test.shape[0]),...]

    # map input data to [0,1]
    x_train = x_train[...,None]/255.
    y_train = y_train[...,None]/255.
    x_test = x_test[...,None]/255.
    y_test = y_test[...,None]/255.
    
    # train model
    model.fit(x=[x_train, y_train], y=y_train, batch_size=256, epochs=1, verbose=1, validation_split=0.5)
    
    i = np.random.randint(x_test.shape[0])
    x = x_test[i:i+1,...]
    y = y_test[i:i+1,...]
    r = model.predict([x,y])
    d = get_sampling_grid([x,y])[0]

    # plot example
    # With no training the network should apply the identity transform to the moving image
    xx, yy = np.meshgrid(range(x.shape[1]), range(x.shape[2]))
    dx, dy = np.squeeze(d[...,0]) + xx, np.squeeze(d[...,1]) + yy
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(x),cmap='gray')
    plt.contour(xx, 50, alpha=0.75, linewidths=0.5)
    plt.contour(yy, 50, alpha=0.75, linewidths=0.5)
    plt.title('Moving Image')
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(r),cmap='gray')
    plt.contour(dx, 50, alpha=0.75, linewidths=0.5)
    plt.contour(dy, 50, alpha=0.75, linewidths=0.5)
    plt.title('Deformed Image')
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(y),cmap='gray')
    plt.title('Reference Image')
    plt.savefig('example-2d-output.png')