import numpy as np

class MultiLR:
    def __init__(self) :
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,x_train,y_train):
        #logic need to be written and understood first!! so pending
        pass
    
    def predict(self,x_test):
        return (np.dot(x_test,self.coef_) + self.intercept_)