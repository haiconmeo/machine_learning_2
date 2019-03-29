

import numpy as np
from keras.preprocessing import image
from keras import models


test_image = image.load_img('meo.jpg', target_size = (150, 150))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
model= models.load_model('traincatdog.h5')
result = model.predict(test_image)
print (result)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)  