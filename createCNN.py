import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential


def create_neural_network(dense_layers, max_layer_size, conv_layers, X, Y, categories_size):
    for dense_layer in dense_layers:
        for layer_size in max_layer_size:
            for conv_layer in conv_layers:
                name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

                model = Sequential()  # linear stack of layers

                model.add(Conv2D(layer_size / 4, (3, 3), input_shape=X.shape[1:]))  # 3x3, skip -1
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Conv2D(layer_size / 2, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer - 2):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())  # from 2D dataset to 1D
                model.add(Dropout(0.5))

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(categories_size))
                model.add(Activation('softmax'))
                model.summary()
                tensorboard = TensorBoard(log_dir="logs/{}".format(name))

                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'],
                              )

                model.fit(X, Y,
                          batch_size=50,
                          epochs=10,
                          validation_split=0.2,
                          callbacks=[tensorboard])
    model.save('CNN.model')
