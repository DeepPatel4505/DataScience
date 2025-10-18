class SimpleLR:
    def __init__(self) :
        self.m = 0
        self.b = 0
        
    def fit(self,x_train,y_train):
        n = len(x_train)
        x_mean = sum(x_train) / n
        y_mean = sum(y_train) / n
        num = 0
        den = 0
        for i in range(n):
            num += (x_train[i] - x_mean) * (y_train[i] - y_mean)
            den += (x_train[i] - x_mean) ** 2
        self.m = num / den
        self.b = y_mean - self.m * x_mean

    def predict(self,x_test):
        return self.m * x_test + self.b