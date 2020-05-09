import numpy as np
import tensorflow as tf
from smoothTransformer import smoothTransformer2D

import matplotlib.pylab as plt


def getModel(moving, reference):

    i = tf.keras.layers.Concatenate(axis=-1)([moving, reference])

    # encoder
    enc1 = tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(i)
    enc1 = tf.keras.layers.LeakyReLU()(enc1)
    enc2 = tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(enc1)
    enc2 = tf.keras.layers.LeakyReLU()(enc2)
    enc3 = tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(enc2)
    enc3 = tf.keras.layers.LeakyReLU()(enc3)
    enc = tf.keras.layers.Concatenate(axis=-1)([i,enc1,enc2,enc3])
    enc = tf.keras.layers.Dropout(0.5)(enc)

    # decoder
    ddec = tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(2, 2), kernel_initializer='he_normal', padding='same')(enc)
    ddec = tf.keras.layers.LeakyReLU()(ddec)
    ddec = tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = tf.keras.layers.LeakyReLU()(ddec)
    ddec = tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = tf.keras.layers.LeakyReLU()(ddec)

    # Affine
    affine = tf.keras.layers.GlobalAveragePooling2D()(enc)
    affine = tf.keras.layers.Dense(9, bias_initializer='zeros', kernel_initializer='zeros')(affine)
    affine = tf.keras.layers.Activation('linear')(affine)                   # this should be linear and initialized to 0 !
    affine = tf.keras.layers.ActivityRegularization(l1=1e-6)(affine)        # affine regularizer

    # Deformable
    deformable = tf.keras.layers.Conv2D(2, (3, 3), padding='same', bias_initializer='zeros', kernel_initializer='zeros')(ddec)
    deformable = tf.keras.layers.Activation('linear')(deformable)               # this should be linear and initialized to 0 !
    deformable = tf.keras.layers.ActivityRegularization(l1=1e-6)(deformable)   # deformable regularizer

    # Smooth transformer block
    [forward, inverse, sampling_grid, sampling_grid_inverse] = smoothTransformer2D(maxgrad=10)([moving, reference, deformable, affine])

    return forward, inverse, sampling_grid, sampling_grid_inverse


if __name__ == '__main__':

    # get model
    moving = tf.keras.layers.Input((28,28,1))
    reference = tf.keras.layers.Input((28,28,1))
    forward, inverse, sampling_grid, sampling_grid_inverse = getModel(moving, reference)
    model = tf.keras.models.Model(inputs=[moving, reference], outputs=[forward, inverse])
    model.compile(optimizer='adam', loss='mse')

    # get data
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    y_train = x_train[np.random.permutation(x_train.shape[0]),...]
    y_test = x_test[np.random.permutation(x_test.shape[0]),...]

    # map input data to [0,1]
    x_train = x_train[...,None]/255.
    y_train = y_train[...,None]/255.
    x_test = x_test[...,None]/255.
    y_test = y_test[...,None]/255.
    
    # train model
    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=256, epochs=10, verbose=1, validation_split=0.5)

    # get inference model
    inference_model = tf.keras.models.Model(inputs=model.input, outputs=[forward, inverse, sampling_grid, sampling_grid_inverse])
    
    i = np.random.randint(x_test.shape[0])
    x = x_test[i:i+1,...]
    y = y_test[i:i+1,...]
    fwd, bwd, grid, grid_inv = inference_model.predict([x,y])
    
    # plot example
    # With no training the network should apply the identity transform to the moving image
    xx, yy = np.meshgrid(range(x.shape[1]), range(x.shape[2]))
    dx, dy = np.squeeze(grid[...,0]) + xx, np.squeeze(grid[...,1]) + yy
    dxi, dyi = np.squeeze(grid_inv[...,0]) + xx, np.squeeze(grid_inv[...,1]) + yy

    plt.figure(figsize=(10,4))
    plt.subplot(1, 4, 1)
    plt.imshow(np.squeeze(x),cmap='gray')
    plt.title('Moving Image')
    plt.subplot(1, 4, 2)
    plt.imshow(np.squeeze(fwd),cmap='gray')
    plt.contour(dx, 50, alpha=0.5, linewidths=0.5)
    plt.contour(dy, 50, alpha=0.5, linewidths=0.5)
    plt.title('Forward Deformation \n applied on Moving Image')
    plt.subplot(1, 4, 3)
    plt.imshow(np.squeeze(bwd),cmap='gray')
    plt.contour(dxi, 50, alpha=0.5, linewidths=0.5)
    plt.contour(dyi, 50, alpha=0.5, linewidths=0.5)
    plt.title('Inverse Deformation \n applied on Deformed Image')
    plt.subplot(1, 4, 4)
    plt.imshow(np.squeeze(y),cmap='gray')
    plt.title('Reference Image')
    plt.tight_layout()
    plt.savefig('example-2d-output.png')