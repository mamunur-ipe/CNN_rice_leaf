"""
@Author: Mamunur Rahman
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import cv2     
import os  

folders = os.listdir('F:/Python Practice/Deep learning/Rice leaf diseases/train_test data/train')
print(folders)

## resize the images, perform augmentattion, and save in designated folders
img_size_x = 500
img_size_y = 100
n_copies = 20   #number of augmented images to be created

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

for folder in folders:
    folder_path_read = os.path.join('train_test data/train', folder)
    folder_path_save_resize = os.path.join('train_test data/train_resized', folder)
    folder_path_save_augmented = os.path.join('train_test data/train_augmented', folder)
    image_names = os.listdir(folder_path_read)
    k=0 #augmented image name prefix
    for image in image_names:
        image_read_path = os.path.join(folder_path_read, image)
        image_save_path_resize = os.path.join(folder_path_save_resize, image)
        img_array = cv2.imread(image_read_path)  # convert to array
        resized_array = cv2.resize(img_array, (img_size_x, img_size_y))
        cv2.imwrite(image_save_path_resize, resized_array) #save resized image
        ##image augmentation
        img = load_img(image_save_path_resize)  # this is a resized image
        x = img_to_array(img)  # this is a Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=folder_path_save_augmented, save_prefix=f'augmented_{k}', save_format='jpeg'):
            k += 1
            i += 1
            if i >= n_copies:
                break  # otherwise the generator would loop indefinitely

#------------------------------------------------------------------------------
folders = os.listdir('train_test data/train_augmented')
print(folders)

dictionary = {'Bacterial leaf blight' : 0,
              'Brown spot' : 1,
              'Leaf smut' : 2
             }
dictionary['Bacterial leaf blight']

# create file path of the images
file_path = []
y = []

for folder in folders:
    folder_path = os.path.join('train_test data/train_augmented', folder)
    image_names = os.listdir(folder_path)
    for image in image_names:
        file_path.append(os.path.join(folder_path, image))
        y.append(dictionary[folder])
    
print( file_path[0])
print(y[0])


data = []   #iamge and label will be saved here
i=0
for path in file_path:  # iterate over each image item
            try:
                img_array = cv2.imread(path)  # convert to array
                data.append([img_array, y[i]])  # add this to our data
#            except Exception as e:  # in the interest in keeping the output clean...
#                pass
            
            except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
            i=i+1

print(len(data))


# pickle the data
import pickle
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
    
#------------------------------------------------------------------------------
# unpickle the data
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)


# shuffle the data
import random
random.shuffle(data)

# print one sample image
print(data[1])
plt.imshow(data[0][0], cmap='gray')

# separate feature and labels and reshape the feature matrix
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# print(X[0].reshape(-1, image_size_x, image_size_y, 1))

X = np.array(X).reshape(-1, img_size_x, img_size_y,3)
print(np.array(X).shape)
y = np.array(y)

## scale the feature vector
X = X/255.0

#------------------------------------------------------------------------------
## build CNN model
model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#Hidden layers
no_of_hidden_layer = 2
for i in range(no_of_hidden_layer):
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

#Output layer
model.add(Dense(3))
model.add(Activation('softmax'))

# compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

#train the model
model.fit(X, y, batch_size=3, epochs=20)
