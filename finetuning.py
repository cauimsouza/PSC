import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import time
import keras
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from sklearn.metrics import log_loss

# Constructing vgg16 net
def vgg16_base_model(img_rows, img_cols, channel=1, num_classes=None):
    
    # Load VGG16 net with weights from ImageNet
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    model = keras.applications.VGG16(weights='imagenet',
                                     include_top=False,
                                     input_tensor=input_tensor)
    
    #### KEY PART ####
    # Create a Top Model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.15))
    top_model.add(Dense(4, activation='sigmoid'))
    
    ##################

    # Create a REAL model
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
        
    # Lock the top conv layers
    for layer in new_model.layers:
        layer.trainable = False    
        
    # Concatenate the two models
    new_model.add(top_model)
        
    # Compile Model
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    new_model.summary()
    return new_model

# Split Data X, y in train set and test set
def split_data(X, y, test_size):
    onehotencoder = preprocessing.LabelEncoder()
    onehotencoder.fit(y)
    y = keras.utils.to_categorical(onehotencoder.transform(y), 4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=42, stratify=y)                                                       
    return X_train, X_test, y_train, y_test
    
    
if __name__ == '__main__':
    
    # Set parameters and constants
    img_rows, img_cols = 64, 431 #Resolution of the spectrogram
    channel = 1
    num_classes = 4
    batch_size = 32
    nb_epoch = 5
    test_size = 0.20  
        
    # Load data
    spectrograms = np.load('data.npy')
    labels = np.load('labels.npy')

    # Preprossing DATA
    y = labels[:, 0]
    X = spectrograms
    X = np.reshape(X, (550, 96, 862))
    for i, matrix in enumerate(X):
        X[i] = X[i] - np.mean(matrix)
        X[i] /= np.std(matrix)
    X = np.reshape(X, (550, 64, 431, 3))

    # Spliting the dataset in traning set and test set
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    
    # Constructing a vgg16 net    
    model = vgg16_base_model(img_rows, img_cols, channel, num_classes=num_classes)
    
    # Definition of the tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_graph=True, write_images=True)

    # Fit Model
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard]
              )

    # Predict Model
    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    score = log_loss(y_test, predictions_valid)
    print(score)

    # SAVING THE MODEL

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    
    