import numpy as np
class GDRegressor:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m =  0.001
        self.b = 0.001
        self.history=[]
    def fit(self, x_train, y_train):
        x_train = x_train.values.ravel()
        y_train = y_train.values.ravel()
        n = len(x_train)
        for i in range(self.epochs):
            # Calculate predictions
            y_pred = self.m * x_train + self.b
            # Calculate gradients
            gradient_b = (-2/n) * np.sum(y_train - y_pred)  # Gradient w.r.t. intercept b
            gradient_m = (-2/n) * np.sum((y_train - y_pred) * x_train)  # Gradient w.r.t. slope m
            # Update parameters
            self.b -= self.learning_rate * gradient_b  # Update intercept b
            self.m -= self.learning_rate * gradient_m  # Update slope m
            self.history.append([self.m,self.b])
    def predict(self, x_test):
        return self.m * x_test + self.b