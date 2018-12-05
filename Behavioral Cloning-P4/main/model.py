from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from random import shuffle
from keras.utils import plot_model
import os


class Model(object):

    def __init__(self,basepath, model_weights=""):
        self.basepath = os.path.dirname(basepath)
        self.model = Sequential()
        # Preprocess incoming data, centered around zero with small standard deviation
        ch, row, col = 3, 80, 320  # Trimmed image format
        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
        self.model.add(Cropping2D(cropping=((71, 25), (0, 0))))
        self.model.add(Conv2D(32, kernel_size=(7,7), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(265, (3, 3), activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(265, (2, 2), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(1))
        self.model.summary()
        #plot_model(self.model, to_file='model.png')

        # continue learing by loading some weights
        if model_weights != "":
            self.model.load_weights(model_weights)
            print("loaded model " + model_weights)


    def train(self, samples, batch_size=32, epochs=3, learning_rate=0.0001):
        def loadImages(batch_sample):
            namecenter = os.path.join(self.basepath, "IMG", batch_sample[0].split('/')[-1])
            nameleft = os.path.join(self.basepath, "IMG", batch_sample[1].split('/')[-1])
            nameright = os.path.join(self.basepath, "IMG", batch_sample[2].split('/')[-1])
            center_image = cv2.imread(namecenter)
            center_image = center_image.reshape((1,) + center_image.shape)
            left_image = cv2.imread(nameleft)
            left_image = left_image.reshape((1,) + left_image.shape)
            right_image = cv2.imread(nameright)
            right_image = right_image.reshape((1,) + right_image.shape)


            # flip each image
            image_flipped = np.fliplr(center_image)
            #print(image_flipped.shape)

            return np.vstack((center_image, image_flipped, left_image, right_image))

        def loadSteering(batch_sample):
            offset = 0.2
            center_angle = float(batch_sample[3])
            return np.vstack((center_angle,-center_angle, center_angle+offset, center_angle - offset))


        def generator(samples, batch_size=batch_size):
            num_samples = len(samples)
            shuffle(samples)

            while 1:  # Loop forever so the generator never terminates
                #shuffle(samples)
                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset + batch_size]

                    images = np.vstack([loadImages(batch_sample) for batch_sample in  batch_samples])
                    angles = np.vstack([loadSteering(batch_sample) for batch_sample in  batch_samples])
                    #images = []
                    #angles = []
                    #for batch_sample in batch_samples:
                    #    name = os.path.join( self.basepath, "IMG", batch_sample[0].split('/')[-1])
                    #    center_image = cv2.imread(name)
                    #    #print(name)
                    #    #print(center_image.shape)

                    #    center_angle = float(batch_sample[1])
                    #    images.append(center_image)
                    #    angles.append(center_angle)

                    #    # flip each image
                    #    image_flipped = np.fliplr(center_image)
                    #    measurement_flipped = -center_angle
                    #    images.append(image_flipped)
                    #    angles.append(measurement_flipped)

                    # X_train = np.array(angles)
                    X_train = images
                    #y_train = np.array(angles)
                    y_train = angles
                    yield sklearn.utils.shuffle(X_train, y_train)

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        # compile and train the model using the generator function
        train_generator = generator(train_samples, batch_size=batch_size)
        validation_generator = generator(validation_samples, batch_size=batch_size)
        optimizer = optimizers.Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=optimizer)

        self.model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                            nb_val_samples=len(validation_samples), nb_epoch=epochs)
        """
        If the above code throw exceptions, try 
        model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
        validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
        """


    def save(self, filename="model.h5"):
        # not working since keras is very sensitive in versions of models
        #self.model.save(filename)
        self.model.save_weights('weights_' + filename)
        json_string = self.model.to_json()
        import json
        with open( filename + ".json", 'w') as outfile:
            json.dump(json_string, outfile)

