import os,shutil
import numpy

dataset_dir ='./dataset'  #thu muc goc
face_dataset_dir = os.path.join(dataset_dir,'face')
dogs_dataset_dir = os.path.join(dataset_dir,'dogs')
base_dir = './dog_and_human' # thu muc se chua tap train
#os.mkdir(base_dir)
train_dir = os.path.join(base_dir,'train') # train folder
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation') # validation folder
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test') #test folder
# os.mkdir(test_dir)
#---------------------------------------------------------------
face_train_dir = os.path.join(train_dir,'face')
# os.mkdir(face_train_dir)
dog_train_dir = os.path.join(train_dir,'dog')
# os.mkdir(dog_train_dir)
#---------------------------------------------------------------
face_validation_dir = os.path.join(validation_dir,'face')
# os.mkdir(face_validation_dir)
dog_validation_dir = os.path.join(validation_dir,'dog')
# os.mkdir(dog_validation_dir)
#---------------------------------------------------------------
face_test_dir = os.path.join(test_dir,'face')
# os.mkdir(face_test_dir)
dog_test_dir = os.path.join(test_dir,'dog')
# os.mkdir(dog_test_dir)
#------------------------------face----------------------------------
fnames = ['face.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames :
	from_file =os.path.join(face_dataset_dir,fname)
	to_file =os.path.join(face_train_dir,fname)
	shutil.copyfile(from_file,to_file)

fnames =['face.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
	from_file = os.path.join(face_dataset_dir,fname)
	to_file = os.path.join(face_validation_dir,fname)
	shutil.copyfile(from_file,to_file)
fnames = ['face.{}.jpg'.format(i) for i in range(1500,2000)]	
for fname in fnames:
	from_file = os.path.join(face_dataset_dir,fname)
	to_file = os.path.join(face_test_dir,fname)
	shutil.copyfile(from_file,to_file)
#----------------------------dog------------------------------------	

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]	
for fname in fnames:
	from_file = os.path.join(dogs_dataset_dir,fname)
	to_file = os.path.join(dog_train_dir,fname)
	shutil.copyfile(from_file,to_file)
	fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]	
for fname in fnames:
	from_file = os.path.join(dogs_dataset_dir,fname)
	to_file = os.path.join(dog_validation_dir,fname)
	shutil.copyfile(from_file,to_file)
	fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]	
for fname in fnames:
	from_file = os.path.join(dogs_dataset_dir,fname)
	to_file = os.path.join(dog_test_dir,fname)
	shutil.copyfile(from_file,to_file)
#--------------------------model---------------------------------------
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)


model.save('faceanddog.h5')  
