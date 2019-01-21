from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from random import shuffle
from keras.utils import plot_model
import tensorflow as tf
import pandas as pd
import os


class Model(object):

    def __init__(self,basepath="", model_weights="model_weights.h5", continueLearning=False):
        self.basepath = basepath
        self.model_weights = model_weights
        self.continueLearning = continueLearning
        self.model = Sequential()

        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(300, 300, 3)))
        self.model.add(Conv2D(32, kernel_size=(7,7), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (7, 7), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (7, 7), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(265, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(1024, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.summary()
        #plot_model(self.model, to_file='model.png')


    @staticmethod
    def loadTrainingData(basepath):
        self.basepath = basepath
        if os.path.isfile(self.basepath):
            print("INFO: found file " + self.basepath)
        else:
            raise FileNotFoundError("there is no file " + self.basepath)

    def loadWeights(self, filename):
        self.model_weights = filename
        self.model.load_weights(self.model_weights)
        print("loaded model " + self.model_weights)
        self.graph = tf.get_default_graph()

    def run(self, epochs = 1, batch_size=16, continueLearning=False, learning_rate=0.0001):
	    rownames=["filename", "label"]
	    df = pd.read_csv(self.basepath ,sep=",", names=rownames)

	    samples = [tuple(x) for x in df.values]
	    print("total " + str(len(samples)) + " rows")

	    if continueLearning and os.path.isfile(self.model_weights):
	    	self.model.load_weights(self.model_weights)
	    

	    self.train(samples, batch_size=batch_size, epochs=epochs, learning_rate = learning_rate)
	    self.model.save_weights(self.model_weights)

    @staticmethod
    def reshapeImage(image):   
            #print(image.shape)
            image = cv2.resize(image, (300, 300)) 
            image = image.reshape((1,) + image.shape)

            return image

    def train(self, samples, batch_size, epochs, learning_rate):

        def loadLabel(label):
            label = int(label)
            return label #np.vstack((label))

        def loadImages(filename):
            #sprint(filename)
            image = cv2.imread(filename)
            return reshapeImage(image)

        def generator(samples, batch_size=batch_size):
            num_samples = len(samples)
            shuffle(samples)

            while 1:  # Loop forever so the generator never terminates
                #shuffle(samples)
                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset + batch_size]

                    images = np.vstack([loadImages(batch_sample[0]) for batch_sample in  batch_samples])
                    labels = np.vstack([loadLabel(batch_sample[1]) for batch_sample in  batch_samples])

                    # X_train = np.array(angles)
                    X_train = images
                    #y_train = np.array(angles)
                    y_train = labels

                    #print(y_train)
                    y_one_hot = to_categorical(y_train, num_classes=3)
                    #print(X_train.shape)
                    #print(y_one_hot.shape)
                    #print(y_one_hot)
                    yield sklearn.utils.shuffle(X_train, y_one_hot)

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        # compile and train the model using the generator function
        train_generator = generator(train_samples, batch_size=batch_size)
        validation_generator = generator(validation_samples, batch_size=batch_size)
        optimizer = optimizers.Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


        self.model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                            nb_val_samples=len(validation_samples), nb_epoch=epochs)

    # predict the class
    def predict(self, image):
        image = self.reshapeImage(image)
        with self.graph.as_default():
            prob = self.model.predict(image)
            y=prob.argmax(axis=-1)
        return y[0] 


