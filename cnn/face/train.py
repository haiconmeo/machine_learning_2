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
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "./train/", 
    x_col='filetrain',
    y_col='categories',
    target_size=(224,224),
    class_mode='binary',
    batch_size=15
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "./train/", 
    x_col='filetrain',
    y_col='categories',
    target_size=(224,224),
    class_mode='binary',
    batch_size=15
)
print (len(train_generator))
MODEL.train_model(train_generator,validation_generator)
