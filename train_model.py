# Importing the Keras libraries and packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
# Step 1 - Building the CNN
# Initializing the CNN
Model = tf.keras.models.Sequential()
# First convolution layer and pooling
Model.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
Model.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
Model.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening the layers
Model.add(Flatten())
# Adding a fully connected layer
Model.add(Dense(units=128, activation='relu'))
Model.add(Dropout(0.40))
Model.add(Dense(units=96, activation='relu'))
Model.add(Dropout(0.40))
Model.add(Dense(units=64, activation='relu'))
Model.add(Dense(units=27, activation='softmax')) # softmax for more than 2
# Compiling the CNN
Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2
# Step 2 - Preparing the train/test data and training the model
Model.summary()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(sz, sz),
                                                 batch_size=10,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(sz , sz),
                                            batch_size=10,
                                              color_mode='grayscale',
                                            class_mode='categorical') 
Model.fit(
        training_set,
        # steps_per_epoch=1284,           # No of images in training set
        epochs=5,
        validation_data=test_set,
        # validation_steps=4268
        )                                       # No of images in test set
# Saving the model
model_json = Model.to_json()
with open("Models/model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
Model.save_weights('Models/model_new.h5')
print('Weights saved')

