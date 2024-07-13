import tensorflow as tf
from tensorflow import keras
from keras import layers

from GAN import GAN
from GANMonitor import GANMonitor
from GANMonitorTest import GANMonitorTest
from dataset import getDatasets

(train, test) = getDatasets()

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

print(discriminator)

latent_dim = 128 # the latent space will be made of 128-dimensional vectors

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128), # same number of coefficients as flatten layer is discriminator
        layers.Reshape((8, 8, 128)), # revert the flatten layer of the encoder
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"), # revert the conv2d layers of the encoder
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
    ]
)

print(generator)

epochs = 10

print('Creating GAN...')
gan = GAN(discriminator=discriminator,
          generator=generator,
          latent_dim=latent_dim)

print('Compiling GAN...')
gan.compile(
    d_optimizer =keras.optimizers.Adam(learning_rate=1e-5),
    g_optimizer =keras.optimizers.Adam(learning_rate=1e-5),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

print(gan.metrics_names)

print('Fitting GAN...')
history = gan.fit(
                     train, epochs=epochs,
                     callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
                 )



gan_history = history.history

d_loss_history = gan_history['d_loss']
g_loss_history = gan_history['g_loss']

with open('train_d_loss.txt', 'w') as fp:
    for item in d_loss_history:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done writing train d loss scores')
    fp.close()

with open('train_g_loss.txt', 'w') as fp:
    for item in g_loss_history:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done writing train g loss scores')
    fp.close()

history = gan.evaluate(
    test,
    callbacks=[GANMonitorTest(num_img=10, latent_dim=latent_dim)]
)
print(history)
gan_history = history.history

d_loss_history = gan_history['test_d_loss']
g_loss_history = gan_history['test_g_loss']

with open('test_d_loss.txt', 'w') as fp:
    for item in d_loss_history:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done writing test d loss scores')
    fp.close()

with open('test_g_loss.txt', 'w') as fp:
    for item in g_loss_history:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done writing test g loss scores')
    fp.close()