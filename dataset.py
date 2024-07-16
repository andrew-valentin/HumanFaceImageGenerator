from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from tensorflow.python.client import device_lib 
import cv2

def createDatasets():
    print('Creating training dataset...')
    training = keras.utils.image_dataset_from_directory(
        "img_align_celeba",
        label_mode=None, # return images not labels
        validation_split=0.2,
        subset="training",
        shuffle=True,
        seed=0,
        image_size=(64, 64),
        batch_size=32,
        smart_resize=True # by using smart_resize we preserve aspect ratio
    )

    print('Creating testing dataset...')
    validation = keras.utils.image_dataset_from_directory(
        "img_align_celeba",
        label_mode=None, # return images not labels
        validation_split=0.2,
        subset="validation",
        shuffle=True,
        seed=0,
        image_size=(64, 64),
        batch_size=32,
        smart_resize=True # by using smart_resize we preserve aspect ratio
    )

    training = training.map(lambda x: x / 255.)
    validation = validation.map(lambda x: x / 255.)

    print('Datasets created.')

    return (training, validation)

def getDatasets():
    print('Getting dataset...')

    (train, test) = createDatasets()

    # for x in dataset:
    #     plt.axis("off")
    #     plt.imshow((x.numpy() * 255).astype("int32")[0])
    #     plt.show()
    #     break

    print('Returning datasets...')
    return (train, test)
