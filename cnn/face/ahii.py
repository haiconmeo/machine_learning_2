from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
#from keras.layers import Dropout,Flatten,MaxPooling2D, Input, ZeroPadding2D,Convolution2D
#-----------------------model vgg------------------------

model = models.Sequential()
model.add(layers.ZeroPadding2D((1,1),input_shape=(224,224,3)))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.ZeroPadding2D((1,1)))
model.add(layers.Convolution2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

model.add(layers.Convolution2D(4096, (7, 7), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Convolution2D(4096, (1, 1), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Convolution2D(2622, (1, 1)))
model.add(layers.Flatten())
model.add(layers.Activation('softmax'))

#--------------load file train and test --------------
import PIL.Image as img
import numpy as np 
import pandas as pd 
import os
import MODEL
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

filetrain = os.listdir('./train')
categories=[]
for f in filetrain:
	tam =f.split('_')[0]
	if tam=='face':
		categories.append(1)
	else :
		categories.append(0)	



df = pd.DataFrame({
	'filetrain':filetrain,
	'categories':categories
	})
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_datagen = ImageDataGenerator(
    rotation_range=40,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./train/", 
    x_col='filetrain',
    y_col='categories',
    
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary'
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./train/", 
    x_col='filetrain',
    y_col='categories',
    
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary'
)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
checkpoint = ModelCheckpoint('trainface.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
