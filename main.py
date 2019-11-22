import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from datetime import datetime
from LReLU import leaky_relu
from ReadNormalizedOptdigitsDataset import ReadNormalizedOptdigitsDataset


batch_size = 64
epochs = 20
num_classes = 10

img_rows, img_cols = 8, 8
x_train, y_train, x_valid, y_valid, x_test, y_test = ReadNormalizedOptdigitsDataset('optdigits_train.txt','optdigits_valid.txt','optdigits_test.txt')


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model1 = Sequential()
model1.add(Conv2D(1, kernel_size=(4,4), activation='linear', input_shape = input_shape))
model1.add(BatchNormalization())
model1.add(Flatten())
model1.add(Activation(leaky_relu))
model1.add(Dense(num_classes, activation='softmax'))

logdir = "./logs/HW4_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# compile the model
model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model1.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, verbose = 1, validation_data=(x_valid, y_valid), callbacks=[tensorboard_callback])
# do the evaluation on test data set
score = model1.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])


print('MODEL 2')
model2 = Sequential()
model2.add(Conv2D(20, kernel_size=(3,3), activation='linear', input_shape = input_shape))
model2.add(BatchNormalization())
model2.add(Activation(leaky_relu))
model2.add(MaxPooling2D(pool_size = (3,3), strides = 1))
model2.add(Conv2D(32, kernel_size=(3,3)))
model2.add(BatchNormalization())
model2.add(Flatten())
model2.add(Activation(leaky_relu))
model2.add(Dense(num_classes, activation='softmax'))
model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

logdir = "./logs/HW4_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# compile the model
model2.fit(x_train, y_train, batch_size = batch_size, epochs=epochs, verbose = 1, validation_data=(x_valid, y_valid), callbacks=[tensorboard_callback])
# do the evaluation on test data set
score = model2.evaluate(x_test, y_test, verbose=0)
