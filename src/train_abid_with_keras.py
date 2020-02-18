from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers

import pandas as pd
import numpy as np
import dataset as ds

img_dir= "/home/jaejun/workspace/abid_challenge/dataset/data/bin-images/"
meta_dir = "/home/jaejun/workspace/abid_challenge/dataset/data/metadata/"

if __name__ == "__main__":

    df = ds.make_counting_df(img_dir,meta_dir, 7)

    train_df = df.sample(frac=0.80, random_state=45)
    test_df = df.copy()
    test_df = test_df.drop(train_df.index)    
    test_label = test_df['label'].values

    datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.20)

    train_generator=datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_dir,
        x_col="id",
        y_col="label",
        subset="training",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(224,224))


    valid_generator=datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=img_dir,
        x_col="id",
        y_col="label",
        subset="validation",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(224,224))


    test_datagen=ImageDataGenerator(rescale=1./255.)
    test_generator=test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=img_dir,
        x_col="id",
        y_col=None,
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(224,224))
