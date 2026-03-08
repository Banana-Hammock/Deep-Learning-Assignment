import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from CustomDL import *
from tensorflow.keras.datasets import mnist

t = 1.0
f = 0.0


# MNIST Digit Classification

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

y_train = one_hot(y_train)
y_test = one_hot(y_test)

model = Automation()

model.add(Layer(128, input_shape=(784,), activation="relu"))
model.add(Layer(64, activation="relu"))
model.add(Layer(10, activation="softmax"))

model.run(loss="crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=64)

pred = model.predict(X_test)

acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))

print("Accuracy:", acc)