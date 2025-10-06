import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(trainX, trainY), (testX, testY) = mnist.load_data()

print("trainX:", trainX.shape)
print("trainY:", trainY.shape)
print("testX:", testX.shape)
print("testY:", testY.shape)

# Display a sample digit
def display_digit(index):
    label = trainY[index]  # trainY is not one-hot, just int
    image = trainX[index].reshape((28, 28))
    plt.title('Sample: %d  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()

display_digit(0)

# Prepare the data
dimension_input = trainX.shape[1] * trainX.shape[2]
trainX_norm = trainX / 255.0
testX_norm = testX / 255.0

trainY_cate = to_categorical(trainY)
testY_cate = to_categorical(testY)

# Build the model
def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(dimension_input,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

model_tofit = build_model()
model_tofit.summary()

# Reshape the data for the model
trainX_norm = trainX_norm.reshape(trainX_norm.shape[0], dimension_input)
testX_norm = testX_norm.reshape(testX_norm.shape[0], dimension_input)

# Train the model
model_tofit.fit(trainX_norm, trainY_cate, validation_data=(testX_norm, testY_cate), epochs=5, batch_size=32, verbose=2)

# Evaluate the model
predictions = np.array(model_tofit.predict(testX_norm)).argmax(axis=1)
actual = testY_cate.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

print("Test accuracy: ", test_accuracy)

confusion_matrix(y_true=testY.argmax(axis=0), y_pred=actual)
print(classification_report(np.argmax(testY_cate, axis=1), predictions))

# Manual MLP implementation for comparison
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def neuron(inputs, weights, bias, activation='relu'):
    z = np.dot(inputs, weights) + bias
    if activation == 'sigmoid':
        return sigmoid(z)
    elif activation == 'relu':
        return relu(z)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.bias = np.zeros(output_dim)
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

class MLP:
    def __init__(self, layer_dims, activations):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(Layer(layer_dims[i], layer_dims[i+1], activations[i]))

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out