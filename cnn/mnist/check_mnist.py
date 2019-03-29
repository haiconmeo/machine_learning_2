import cv2
import numpy as np
from keras import models
from keras.preprocessing import image
test_image = image.load_img('so8.jpg', target_size = (28, 28),grayscale=True)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print (test_image.shape)
model= models.load_model('train.h5')
result = model.predict(test_image)
for i in range(10):
	if result[0][i] !=0:
		print (i)

		break
	
"""
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)    
img=cv2.imread('so8.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=img.astype('float32')/255
img =cv2.resize(img,(28,28))
print (img)
model= models.load_model('train.h5')
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
result = model.predict(np.expand_dims(img, axis=0))
print(result[0][0])"""