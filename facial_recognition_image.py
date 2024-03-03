import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models,layers,datasets
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

augmentor=ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,validation_split=0.2)
augmented_trained_data=augmentor.flow_from_directory("emotions_database",target_size=(150,150),batch_size=32,color_mode="rgb",class_mode="categorical")
augmented_testing_data=augmentor.flow_from_directory("emotions_database",target_size=(150,150),batch_size=32,color_mode="rgb",class_mode="categorical")
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),layers.MaxPool2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Conv2D(128,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dropout(0.2),layers.Dense(5,activation="softmax")])
model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),metrics=["accuracy"])
model.fit(augmented_trained_data,epochs=20)
model.save("emotions.keras")
test_loss,test_accuracy=model.evaluate(augmented_testing_data)
print(test_loss,test_accuracy)

all_images=[]
all_labels=[]
open_emotion_database=os.listdir("emotions_database")
for i in open_emotion_database:
    sub_folder_path=os.path.join("emotions_database",i)
    sub_folder=os.listdir(sub_folder_path)
    for j in sub_folder:
        image_path=os.path.join("emotions_database",i,j)
        image_access=image.load_img(image_path,target_size=(150,150))
        image_array=image.img_to_array(image_access)
        normalized_images=image_array/255
        all_images.append(normalized_images)
        all_labels.append(i)
print("preprocessing of data is done")
numpy_images=np.array(all_images)
numpy_labels=np.array(all_labels)
label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(numpy_labels)
number_of_labels=len(label_encoder.classes_)
one_hot_encoded_table=to_categorical(encoded_labels,number_of_labels)

X=numpy_images
Y=one_hot_encoded_table
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),layers.MaxPool2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Conv2D(128,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dropout(0.2),layers.Dense(number_of_labels,activation="softmax")])
model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),metrics=["accuracy"])
print("now model is training")
model.fit(X_train,Y_train,epochs=20)
model.save("emotions.keras")
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)
print(test_loss,test_accuracy)

import os
import tensorflow as tf
from tensorflow.keras import models,layers,datasets
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

all_images=[]
all_labels=[]
open_emotion_database=os.listdir("emotions_database")
for i in open_emotion_database:
    path_sub_folder=os.path.join("emotions_database",i)
    accessing_sub_folder=os.listdir(path_sub_folder)
    for j in accessing_sub_folder:
        path_image=os.path.join("emotions_database",i,j)
        image_load=image.load_img(path_image,target_size=(150,150))
        image_array=image.img_to_array(image_load)
        normalized_images=image_array/255
        all_images.append(normalized_images)
        all_labels.append(i)
numpy_images=np.array(all_images)
numpy_labels=np.array(all_labels)

label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(numpy_labels)
number_of_labels=len(label_encoder.classes_)
one_hot_encoded_table=to_categorical(encoded_labels,number_of_labels)
X=numpy_images
Y=one_hot_encoded_table
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),layers.MaxPool2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Conv2D(128,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dropout(0.2),layers.Dense(number_of_labels,activation="softmax")])
model.compile(optimizer="adam",loss=tf.keras.losses.Categorical.Crossentropy(from_logits=False),metrics=["accuracy"])
model.fit(X_train,Y_train)
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)

