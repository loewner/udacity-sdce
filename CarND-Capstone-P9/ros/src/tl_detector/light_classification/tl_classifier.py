from styx_msgs.msg import TrafficLight
import model as md

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from random import shuffle
from keras.utils import plot_model
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.model = md.Model("./")
        self.model.loadWeights("light_classification/model_weights.h5")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return self.model.predict(image)
