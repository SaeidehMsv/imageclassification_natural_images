import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout


 



def create_model():
  model = Sequential()
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(256, 256, 3)))
  model.add(MaxPooling2D(pool_size=(2, 2),padding="valid", strides=None ))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding="valid", strides=None ))
  model.add(Dropout(0.5))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding="valid", strides=None ))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding="valid", strides=None ))
  model.add(Conv2D(100, (3, 3), activation='relu', kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2, 2),padding="valid", strides=None ))
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(50, activation = 'swish'))
  model.add(Dense(50, activation='swish'))
  model.add(Dropout(0.5))
  model.add(Dense(30, activation='swish'))
  model.add(Dropout(0.5))
  model.add(Dense(8, activation='softmax'))
  model.compile(Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
  return model  