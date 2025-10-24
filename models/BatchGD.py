import numpy as np

class BatchGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coeff_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.coeff_ = np.ones(n_features)
        self.intercept_ = 0

        for _ in range(self.epochs):
            y_predicted = np.dot(X_train, self.coeff_) + self.intercept_

            dw = (-2/n_samples) * np.dot(X_train.T, (y_train - y_predicted))
            db = (-2/n_samples) * np.sum(y_train - y_predicted)

            self.coeff_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

    def predict(self, X_test):
        return np.dot(X_test, self.coeff_) + self.intercept_