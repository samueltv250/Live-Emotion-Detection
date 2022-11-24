
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.python.keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.python.keras import regularizers







model = "inputs/models/modelV2.h5"
model = load_model(model)

train_dir = "inputs/train"
test_dir = "inputs/test"
batch_szi = 64


img_size = 48
train_datagen = ImageDataGenerator(      width_shift_range = 0.1,
                                         height_shift_range = 0.1,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         validation_split = 0.2
                                        )

train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                    target_size = (img_size,img_size),
                                                    batch_size = 64,
                                                    shuffle=False,
                                                    color_mode = "grayscale",
                                                    class_mode = "categorical",
                                                    subset = "training"
                                                   )



validation_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.25)

validation_generator = validation_datagen.flow_from_directory( directory = test_dir,
                                                              target_size = (img_size,img_size),
                                                              batch_size = 64,
                                                              shuffle=False,
                                                              color_mode = "grayscale",
                                                              class_mode = "categorical",
                                                              subset = "validation"
                                                             )



Y_pred = model.predict(validation_generator )
y_pred = np.argmax(Y_pred, axis=1)

results = confusion_matrix(validation_generator.classes, y_pred) 

print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(validation_generator.classes, y_pred)) 
print ('Report : ')
target_names = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

print (classification_report(validation_generator.classes, y_pred, target_names=target_names))



















img = tf.keras.utils.load_img("inputs/test/sad/im100.png",target_size = (48,48),color_mode = "grayscale")
img = np.array(img)
plt.imshow(img)
print(img.shape)


label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
img = img.reshape(1,48,48,1)
result = model.predict(img)



result = list(result[0])
print(result)
img_index = result.index(max(result))
print(label_dict[img_index])
plt.show()