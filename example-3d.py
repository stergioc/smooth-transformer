import numpy as np
import tensorflow as tf
from smoothTransformer import smoothTransformer3D

import matplotlib.pylab as plt

def getModel(moving, reference):

    i = tf.keras.layers.Concatenate(axis=-1)([moving, reference])

    # encoder
    enc1 = tf.keras.layers.Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(i)
    enc1 = tf.keras.layers.LeakyReLU()(enc1)
    enc2 = tf.keras.layers.Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(enc1)
    enc2 = tf.keras.layers.LeakyReLU()(enc2)
    enc3 = tf.keras.layers.Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), kernel_initializer='he_normal', padding='same')(enc2)
    enc3 = tf.keras.layers.LeakyReLU()(enc3)
    enc = tf.keras.layers.Concatenate(axis=-1)([i,enc1,enc2,enc3])
    enc = tf.keras.layers.Dropout(0.5)(enc)

    # decoder
    ddec = tf.keras.layers.Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), kernel_initializer='he_normal', padding='same')(enc)
    ddec = tf.keras.layers.LeakyReLU()(ddec)
    ddec = tf.keras.layers.Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = tf.keras.layers.LeakyReLU()(ddec)
    ddec = tf.keras.layers.Conv3D(32, (3, 3, 3), dilation_rate=(1, 1, 1), kernel_initializer='he_normal', padding='same')(ddec)
    ddec = tf.keras.layers.LeakyReLU()(ddec)

    # Affine
    affine = tf.keras.layers.GlobalAveragePooling3D()(enc)
    affine = tf.keras.layers.Dense(12, bias_initializer='zeros', kernel_initializer='zeros')(affine)
    affine = tf.keras.layers.Activation('linear')(affine)                   # this should be linear and initialized to 0 !
    affine = tf.keras.layers.ActivityRegularization(l1=1e-7)(affine)        # affine regularizer

    # Deformable
    deformable = tf.keras.layers.Conv3D(3, (3, 3, 3), padding='same', bias_initializer='zeros', kernel_initializer='zeros')(ddec)
    deformable = tf.keras.layers.Activation('linear')(deformable)               # this should be linear and initialized to 0 !
    deformable = tf.keras.layers.ActivityRegularization(l1=1e-7)(deformable)    # deformable regularizer

    [forward, inverse, sampling_grid, sampling_grid_inverse] = smoothTransformer3D(maxgrad=10)([moving, reference, deformable, affine])
    return forward, inverse, sampling_grid, sampling_grid_inverse

def getMnist3D(ntrain=2500, ntest=500, random_yawn=False):
    def normalization(x):
        return 255* ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-7))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_3d = np.pad(np.repeat(x_train[:ntrain,...,None],6,axis=-1), ((0,0),(0,0),(0,0),(11,11)))
    x_test_3d = np.pad(np.repeat(x_test[:ntest,...,None],6,axis=-1), ((0,0),(0,0),(0,0),(11,11)))

    if random_yawn:
        from scipy.ndimage import affine_transform
        for i in range(x_train_3d.shape[0]):
            theta = 4*np.random.random()-2
            tra = [[1,0,0,-14],[0,1,0,-14],[0,0,1,-14],[0,0,0,1]]
            rot = [[np.cos(theta), 0, np.sin(theta),0],[0,1,0,0],[-np.sin(theta), 0, np.cos(theta),0],[0,0,0,1]]
            trainv = [[1,0,0,14],[0,1,0,14],[0,0,1,14],[0,0,0,1]]
            affine = affine = np.matmul(trainv,np.matmul(rot, tra))
            x_train_3d[i,...] = normalization(affine_transform(x_train_3d[i,...], affine))

        for i in range(x_test_3d.shape[0]):
            theta = 4*np.random.random()-2
            tra = [[1,0,0,-14],[0,1,0,-14],[0,0,1,-14],[0,0,0,1]]
            rot = [[np.cos(theta), 0, np.sin(theta),0],[0,1,0,0],[-np.sin(theta), 0, np.cos(theta),0],[0,0,0,1]]
            trainv = [[1,0,0,14],[0,1,0,14],[0,0,1,14],[0,0,0,1]]
            affine = affine = np.matmul(trainv,np.matmul(rot, tra))
            x_test_3d[i,...] = normalization(affine_transform(x_test_3d[i,...], affine))

    return (x_train_3d, y_train[:ntrain]), (x_test_3d, y_test[:ntest])

if __name__ == '__main__':

    # input
    moving = tf.keras.layers.Input((28,28,28,1))
    reference = tf.keras.layers.Input((28,28,28,1))
    forward, inverse, sampling_grid, sampling_grid_inverse = getModel(moving, reference)
    model = tf.keras.models.Model(inputs=[moving, reference], outputs=[forward,inverse])
    model.compile(optimizer='adam', loss='mse')

    # generate a simple 3D mnist dataset
    (x_train, _), (x_test, _) = getMnist3D(random_yawn=False)

    y_train = x_train[np.random.permutation(x_train.shape[0]),...]
    y_test = x_test[np.random.permutation(x_test.shape[0]),...]

    # map input data to [0,1]
    x_train = x_train[...,None]/255.
    y_train = y_train[...,None]/255.
    x_test = x_test[...,None]/255.
    y_test = y_test[...,None]/255.

    # train model
    model.fit(x=[x_train, y_train], y=[y_train, x_train], batch_size=64, epochs=20, verbose=1, validation_split=0.5)
    
    # get inference model
    inference_model = tf.keras.models.Model(inputs=model.input, outputs=[forward, inverse, sampling_grid, sampling_grid_inverse])

    i = np.random.randint(x_test.shape[0])
    x = x_test[i:i+1,...]
    y = y_test[i:i+1,...]
    fwd, bwd, grid, grid_inv = inference_model.predict([x,y])
    
    # plot example
    # With no training the network should apply the identity transform to the moving image
    xx, yy, zz = np.meshgrid(range(x.shape[1]), range(x.shape[2]), range(x.shape[3]), indexing='ij')
    dx, dy, dz = grid[0,:,:,:,0] + xx, grid[0,:,:,:,1] + yy, grid[0,:,:,:,1] + zz
    dxi, dyi, dzi = grid_inv[0,:,:,:,0] + xx, grid_inv[0,:,:,:,1] + yy, grid_inv[0,:,:,:,1] + zz
    
    plt.figure(figsize=(10,4))
    plt.subplot(1, 4, 1)
    plt.imshow(np.squeeze(x)[:,:,14],cmap='gray')
    plt.title('Moving Image')
    plt.subplot(1, 4, 2)
    plt.imshow(np.squeeze(fwd)[:,:,14],cmap='gray')
    plt.contour(dx[...,14], 50, alpha=0.5, linewidths=0.5)
    plt.contour(dy[...,14], 50, alpha=0.5, linewidths=0.5)
    plt.title('Forward Deformation \n applied on Moving Image')
    plt.subplot(1, 4, 3)
    plt.imshow(np.squeeze(bwd)[:,:,14],cmap='gray')
    plt.contour(dxi[...,14], 50, alpha=0.5, linewidths=0.5)
    plt.contour(dyi[...,14], 50, alpha=0.5, linewidths=0.5)
    plt.title('Inverse Deformation \n applied on Reference Image')
    plt.subplot(1, 4, 4)
    plt.imshow(np.squeeze(y)[:,:,14],cmap='gray')
    plt.title('Reference Image')
    plt.tight_layout()
    plt.savefig('example-3d-output.png')

