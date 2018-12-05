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

# train the model
run("../data/runAllInOne2/driving_log.csv", continueLearning=False, epochs=4)
# do some kind of transfer learning
run("../data/data_udacity/driving_log.csv", continueLearning=False, epochs=1)

