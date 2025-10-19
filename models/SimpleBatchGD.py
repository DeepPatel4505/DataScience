import numpy as np

class SimpleBatchGD:
    def __init__(self,learning_rate = 0.1,epochs = 100):
        self.m = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self,x_train,y_train):
        n = len(x_train)
        for i in range(self.epochs):
            loss_m = (-2/n) * np.sum((y_train - self.m*x_train.ravel() - self.b) * x_train.ravel())
            loss_b = (-2/n)* np.sum(y_train - self.m*x_train.ravel() - self.b)
            
            self.m = self.m - (self.learning_rate * loss_m)
            self.b = self.b - (self.learning_rate * loss_b)
        print(self.m,self.b)
    
    def predict(self,x_test):
        return self.m * x_test + self.b 