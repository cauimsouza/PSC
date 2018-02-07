import numpy as np
from time import time
import keras
#from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from sklearn.metrics import log_loss
import splitdata as sp
from sklearn.metrics import confusion_matrix
import pandas as pd

# Constructing vgg16 net
def vgg16_base_model(img_rows, img_cols, channel=1, num_classes=None):
    
    # Load VGG16 net with weights from ImageNet
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    model = keras.applications.VGG16(weights='imagenet',
                                     include_top=False,
                                     input_tensor=input_tensor)
    
    #### KEY PART ####
    # Create a Top Model
    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(128, activation='relu'))
    # top_model.add(Dropout(0.15))
    # top_model.add(Dense(4, activation='sigmoid'))

    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(4, activation='softmax'))
    
    ##################

    # Create a REAL model
    new_model = Sequential()
    for l in model.layers:
        new_model.add(l)
    
    # Pop the last layer
    new_model.layers.pop()
    # new_model.outputs = [new_model.layers[-1].output]
    # new_model.layers[-1].outbound_nodes = []

    for layer in new_model.layers:
        layer.trainable = False 


    # Using just one final layer: number of parameters 53252

    new_model.add(Flatten(input_shape=new_model.output_shape[1:]))
    new_model.add(Dense(4, activation='softmax'))

    # Lock the top conv layers
    # for layer in new_model.layers[:12]:
    #     layer.trainable = False    
        
    # Concatenate the two models
    #new_model.add(top_model)
        
    # Compile Model
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
    new_model.summary()
    return new_model   

if __name__ == '__main__':
    
    # Set parameters and constants
    img_rows, img_cols = 64, 431 #Resolution of the spectrogram
    channel = 1
    num_classes = 4
    batch_size = 32
    nb_epoch = 1
    test_size = 0.20  
    
    X_train, X_test, y_train, y_test, classes = sp.split_data(homo=True)
        
    # Constructing a vgg16 net    
    model = vgg16_base_model(img_rows, img_cols, channel, num_classes=num_classes)
    
    # Definition of the tensorboard
    #tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_graph=True, write_images=True)

    # Fit Model
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, y_test)
             # callbacks=[tensorboard]
              )

    # Predict Model
    predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    score = log_loss(y_test, predictions_valid)
    print(score)
    
    predictions_valid = np.asarray(predictions_valid)
    predictions_valid = np.argmax(predictions_valid, axis=1)
    y_test = np.argmax(y_test, axis=1)

    confmatrix = confusion_matrix(predictions_valid, y_test)
    
    np.save('confusion_matrix.npy', confmatrix)
    
    # SAVING THE MODEL

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_homo.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_homo.h5")
    print("Saved model to disk")
    