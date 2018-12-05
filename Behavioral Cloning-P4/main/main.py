import pandas as pd
import model as md

# load the driving log data
path = "../data/run_small/driving_log.csv"

def run(path, epochs = 1, batch_size=16, model_filename = "model.h5", continueLearning=True, learning_rate=0.0001):
    rownames=["center_path", "left_path", "right_path", "steering_angle", "throttle", "break", "speed"]
    df = pd.read_csv(path,sep=",", names=rownames)

    samples = df.loc[:,["center_path", "left_path", "right_path", "steering_angle"]]
    samples = [tuple(x) for x in samples.values]

    if continueLearning:
        model = md.Model(path, model_weights= "weights_" + model_filename)
    else:
        model = md.Model(path)


    model.train(samples, batch_size=batch_size, epochs=epochs, learning_rate = learning_rate)

    model.save(filename=model_filename)



#run("../data/run_small2/driving_log.csv", continueLearning=False,  epochs=2)
#run("../data/run_small3/driving_log.csv", continueLearning=True)
run("../data/runAllInOne2/driving_log.csv", continueLearning=True, epochs=1)
#run("../data/run_diff1/driving_log.csv", continueLearning=True, epochs=1, learning_rate=0.000003)

#run("../data/run_1lap/driving_log.csv", continueLearning=True, epochs=3)
#run("../data/run_1lap_inv/driving_log.csv", continueLearning=True, epochs=3)


# Split the data
#X_train, y_train = data['features'], data['labels']

#from keras.models import Sequential, Model
#from keras.layers import Cropping2D
#import cv2
#
## set up cropping2D layer
#model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
#...
#
## preprocess data
#
#X_normalized = np.array(X_train / 255.0 - 0.5 )
#
#from sklearn.preprocessing import LabelBinarizer
#label_binarizer = LabelBinarizer()
#y_one_hot = label_binarizer.fit_transform(y_train)
#
## compile and fit the model
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_one_hot, epochs=30, validation_split=0.2)
#
## evaluate model against the test data
#with open('small_test_traffic.p', 'rb') as f:
#    data_test = pickle.load(f)
#
#X_test = data_test['features']
#y_test = data_test['labels']
#
## preprocess data
#X_normalized_test = np.array(X_test / 255.0 - 0.5 )
#y_one_hot_test = label_binarizer.fit_transform(y_test)
#
#print("Testing")
#
#metrics = model.evaluate(X_normalized_test, y_one_hot_test)
#for metric_i in range(len(model.metrics_names)):
#    metric_name = model.metrics_names[metric_i]
#    metric_value = metrics[metric_i]
#    print('{}: {}'.format(metric_name, metric_value))