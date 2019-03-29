import numpy as np
from keras.preprocessing import image
from keras import models


test_image = image.load_img('face.jpg', target_size = (150, 150))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
model= models.load_model('faceanddog.h5')
result = model.predict(test_image)
print (result)
if result[0][0] == 0:
    prediction = 'dog'
else:
    prediction = 'face'
print(prediction)  