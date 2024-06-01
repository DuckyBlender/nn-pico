import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

def clean_data(data):
    (X_train, Y_train), (X_test, Y_test) = data

    X_train = X_train.reshape(60000, 784) 
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train = np.where(X_train >= 0.5, 1, 0).astype('float32')
    X_test = np.where(X_test >= 0.5, 1, 0).astype('float32')

    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)
    
    print(Y_train[0])

    return X_train, Y_train, X_test, Y_test

def train_new_model(model_name, layer_size):
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.Sequential(
        [
            keras.Input(shape=(784,)), # input layer
            layers.Dense(layer_size), # hidden layer
            layers.Activation('sigmoid'),
            layers.Dense(10), # output layer
            layers.Activation('softmax')
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_split=0.1)
    model.save(model_name)

    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])
    return loss_and_metrics[1]



def main():
    model_name = f'mnist_model_10.keras'
    # train_new_model(model_name, 10)
    # clean_data(keras.datasets.mnist.load_data())
    model = keras.models.load_model(model_name)
    
    # Save input layer
    with open('weights.rs', 'w') as f:
        f.write("pub const INPUT_WEIGHTS: [f32; 7840] = [")
        weights = model.layers[0].get_weights()[0].flatten()
        for weight in weights:
            f.write(str(weight))
            f.write(', ')
        f.write('];\n\n')
        f.write("pub const INPUT_BIASES: [f32; 10] = [")
        biases = model.layers[0].get_weights()[1]
        for bias in biases:
            f.write(str(bias))
            f.write(', ')
        f.write('];\n\n')
        f.write("pub const HIDDEN_WEIGHTS: [f32; 100] = [")
        weights = model.layers[2].get_weights()[0].flatten()
        for weight in weights:
            f.write(str(weight))
            f.write(', ')
        f.write('];\n\n')
        f.write("pub const HIDDEN_BIASES: [f32; 10] = [")
        biases = model.layers[2].get_weights()[1]
        for bias in biases:
            f.write(str(bias))
            f.write(', ')
        f.write('];\n')

if __name__ == '__main__':
    main()