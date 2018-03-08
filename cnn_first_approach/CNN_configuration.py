from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#Alert that shows if the CPU or GPU are working
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def _preprocessing(training_file, validation_file):
    """
       Transforming the images into vectors
       :param: training_file, folder in which all the trining images are, splitted into subfolders that indicates their classes
       :param: validation_file, folder in which all the validation images are, splitted into subfolders that indicates their classes
       :return: Both vectors that represent the images
   """
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    training_set = train_datagen.flow_from_directory(training_file, target_size=(64, 64), batch_size=32, class_mode='categorical')
    validation_set = validation_datagen.flow_from_directory(validation_file, target_size=(64, 64), batch_size=32, class_mode='categorical')
    return training_set, validation_set

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

def _traning(classifier, training_set, validation_set, file):
    """
       Training the CNN with the given datasets and storing the weights in the given file
       :param: classifier, CNN already compiled
       :param: trainin_set, ImageGenerator object of the images used for traning the CNN
       :param: validation_set, ImageGenerator object of the images used to test the CNN in every epoch
       :param: file, name of the file in which the weight of the CNN are going to be stored
       :return: CNN already trained
   """

    classifier.fit_generator(training_set, steps_per_epoch=3000, epochs=12, validation_data=validation_set, validation_steps=800)
    classifier.save_weights(file)
    return classifier

def _main():

    #Image preprocessing
    training_set, validation_set = _preprocessing('CaptionTraining2018_ALL/training_set', 'CaptionTraining2018_ALL/validation_set')
    #Defining and compiling the model
    model = _sequential_model()
    model = _compilation(model)
    #Traning the CNN
    _traning(model, training_set, validation_set, '3000_800_12_2.h5')


# Necessary to execute the Main function
if __name__ == "__main__":
     _main()

















