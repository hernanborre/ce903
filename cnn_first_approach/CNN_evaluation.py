from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math

#Alert that shows if the CPU or GPU are working
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def _preprocessing(test_file):
    """
       Transforming the images into vectors
       :param: training_file, folder in which all the trining images are, splitted into subfolders that indicates their classes
       :param: validation_file, folder in which all the validation images are, splitted into subfolders that indicates their classes
       :return: Both vectors that represent the images
   """
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(test_file, target_size=(64, 64), batch_size=32, class_mode='categorical')
    return test_set

def _sequential_model():
    """
        Defining the structure of the CNN that is going to be based on layers and receive images as inputs
        :return: CNN already defined
    """

    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 50, activation = 'relu'))
    classifier.add(Dense(units = 101, activation = 'softmax'))
    return classifier

def _compilation(classifier):
    """
        Chosing the activation function, the stochastic gradient descent algorithm and the performance metric
        :param: classifier, CNN already defined and ready for its compilation
        :return: CNN already compiled
    """

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def _evaluation(classifier, test_set, file):
    """
       Evaluating the accuracy of the CNN with an unseen data set
       :param: classifier, CNN already compiled
       :param: test_set, ImageGenerator object of the images used for testing the CNN
       :param: file, name of the file in which the weight of the CNN are going to be stored
       :return: accuracy of the CNN
   """
    classifier.load_weights(file)
    scores = classifier.evaluate_generator(test_set, steps=100)
    return scores

def softmax(z):
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    return [i / sum_z_exp for i in z_exp]

def _prediction(classifier):
    """
         Perform predictions for given images in the tranined model
         :param: classifier, CNN already prepared
     """
    prediction_datagen = ImageDataGenerator(rescale=1. / 255)
    prediction_set = prediction_datagen.flow_from_directory('prediction_set', target_size=(64, 64), batch_size=32, class_mode=None)
    results = classifier.predict_generator(prediction_set)
    predictions = np.argmax(results, axis=-1)  # multiple categories

    print("Results")
    print(softmax(results[0]))
    print(softmax(results[1]))
    print(softmax(results[2]))
    print(softmax(results[3]))
    print(predictions)

def _main():

    #Image preprocessing
    test_set = _preprocessing('CaptionTraining2018_ALL/test_set')
    #Defining and compiling the model
    model = _sequential_model()
    model = _compilation(model)
    #Evaluating the accuracy of the CNN
    scores = _evaluation(model, test_set, '3000_800_12.h5')
    print("[0] %s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("[1] %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    #Predicting new images
    _prediction(model)



# Necessary to execute the Main function
if __name__ == "__main__":
     _main()

















