from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
#from keras.layers import Dropout,Flatten,MaxPooling2D, Input, ZeroPadding2D,Convolution2D
#-----------------------model vgg------------------------
def train_model(training_set,test_set):
	model = models.Sequential()
	model.add(layers.ZeroPadding2D((1,1),input_shape=(224,224, 3)))
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
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	checkpoint = ModelCheckpoint('trainface.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	history = model.fit_generator(training_set,
                         steps_per_epoch = 50,
                         epochs = 15,
                         validation_data = test_set,
                         validation_steps = 50,
                         callbacks = [checkpoint])

#--------------load file train and test --------------


