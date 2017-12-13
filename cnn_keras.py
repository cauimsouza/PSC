import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD

batch_size    = 4
learning_rate = 0.003
momentum=0.9
n_epoch       = 50
n_samples     = 10                              
cv_split      = 0.8                             
train_size    = int(n_samples * cv_split)                               
test_size     = n_samples - train_size

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 1366, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=learning_rate, decay=1e-6, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
score = model.evaluate(x_test, y_test, batch_size=batch_size)