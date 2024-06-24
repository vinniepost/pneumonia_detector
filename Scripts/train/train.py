from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from prefect import task, flow

import logging

logger = logging.getLogger(__name__)


@task
def create_cnn(input_shape=[1024, 1024, 3]):
    """Create the cnn model"""
    print("Creating the cnn model")
    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=2, strides=2))
    cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=2, strides=2))
    cnn.add(Flatten())
    cnn.add(Dense(units=256, activation='relu'))
    cnn.add(Dense(units=256, activation='relu'))
    cnn.add(Dense(units=256, activation='relu'))
    cnn.add(Dense(units=1, activation='sigmoid'))
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Created the cnn model")
    return cnn

@task
def train_cnn(cnn, train_data, test_data, epochs=10):
    """Train the cnn model"""
    print("Training the model")
    # Log the start of training
    logger.info("Training the model")
    cnn.fit(train_data, validation_data=test_data, epochs=epochs)
    return cnn

@flow
def main_train_flow(train_data, test_data, input_shape=[1024, 1024, 3], epochs=10): # get input_shape from load.py
    """Main flow for training the cnn"""
    cnn = create_cnn()
    trained_cnn = train_cnn(cnn, train_data, test_data)
    return trained_cnn
