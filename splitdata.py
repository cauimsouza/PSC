import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras

# Split Data X, y in train set and test set
def split_data(homo):
    # Set parameters and constants
    new_img_rows, new_img_cols = 64, 431 #Resolution of the spectrogram
    img_rows, img_cols = 96, 862
    channel = 3
    num_classes = 4
    test_size = 0.20
    num = 28
        
    # Load data
    X = np.load('data.npy') # Spectrograms
    labels = np.load('labels.npy')

    # Preprossing DATA
    y = labels[:, 0]
    X = np.reshape(X, (550, img_rows, img_cols))
    
    # Homogenous dataset -> 28 examples of each country
    if homo == True:
        X = X[51:]
        y = y[51:]
        X = np.concatenate((X[:56], X[442:]))
        y = np.concatenate((y[:56], y[442:]))   
    # Normalize Data with Standard Scaler
    scaler = preprocessing.StandardScaler()
    X = np.array( [scaler.fit_transform(matrix) for matrix in X] )

    # # Normalize Data    
    # for i, matrix in enumerate(X):
    #     X[i] = X[i] - np.mean(matrix)
    #     X[i] /= np.std(matrix)

    # resize acoxambrado 
    X = np.reshape(X, (X.shape[0], new_img_rows, new_img_cols, channel))

    # OneHotEncode y labels
    onehotencoder = preprocessing.LabelEncoder()
    onehotencoder.fit(y)
    classes = onehotencoder.classes_
    y = keras.utils.to_categorical(onehotencoder.transform(y), num_classes)
    
    # Split the Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=42, stratify=y)                                                       
    return X_train, X_test, y_train, y_test, onehotencoder.classes_


if __name__ == '__main__':
    split_data(False) 