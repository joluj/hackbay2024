from tensorflow import keras
from keras.preprocessing import image
import numpy as np

img_height = 180
img_width = 180

model = keras.models.load_model('model.keras')

rose = './flower_photos/roses/12240303_80d87f77a3_n.jpg'
daisy = 'flower_photos/daisy/5794839_200acd910c_n.jpg'

img = image.load_img(rose, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
predict_x = model.predict(x)
classes_x = np.argmax(predict_x, axis=1)
print(classes_x)

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print("Predicted class: " + class_names[classes_x[0]])
