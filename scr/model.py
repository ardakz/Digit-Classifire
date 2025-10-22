# scr/model.py
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense


def build_lenet(input_shape=(28,28,1),num_classes=10):

    model = Sequential()

    # C1 Convolutional Layer
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding='same'))

    # S2 Pooling Layer
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # C3 Convolutional Layer
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

    # S4 Pooling Layer
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # C5 Fully Connected Convolutional Layer
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

    #Flatten the CNN output so that we can connect it with fully connected layers
    model.add(Flatten())

    # FC6 Fully Connected Layer
    model.add(Dense(84, activation='tanh'))

    # Output Layer with softmax activation
    model.add(Dense(10, activation='softmax'))
    # Model compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    m = build_lenet()
    m.summary()