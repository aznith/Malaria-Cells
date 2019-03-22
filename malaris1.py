import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("cell_images"))
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import fnmatch
import keras
from time import sleep
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation
from keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as k

imagePatches_0 = glob('cell_images/Uninfected/*.png', recursive=True)
imagePatches_1 = glob('cell_images/Parasitized/*.png', recursive=True)
print(len(imagePatches_0))
print(len(imagePatches_1))