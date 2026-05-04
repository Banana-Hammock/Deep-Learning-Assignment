import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([0,1,1,1])

class SimpleNeuron:
  def __init__(self, lr=0.1, epochs=500):
    self.lr = lr
    self.epochs = epochs

  def activate(self, z):
    return 1 if z >= 0 else 0

  def train(self, X, Y):
    self.w = np.zeros(X.shape[1])
    self.b = 0

    for _ in range(self.epochs):
      for i in range(len(X)):
        z = np.dot(X[i], self.w) + self.b
        y_pred = self.activate(z)

        error = Y[i] - y_pred

        # weight update rule
        self.w += self.lr * error * X[i]
        self.b += self.lr * error

  def predict(self, X):
    results = []
    for x in X:
        z = np.dot(x, self.w) + self.b
        results.append(self.activate(z))
    return np.array(results)


model = SimpleNeuron()
model.train(X, Y)

pred = model.predict(X)
accuracy = np.mean(pred == Y)

print("Predictions:", pred)
print("Accuracy:", accuracy)