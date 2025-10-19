import numpy as np

class MultiLR:
    def __init__(self) :
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,x_train,y_train):
        x_train = np.insert(x_train,0,1,axis=1)
        
        # Calculating coefficients using the Normal Equation
        betas = np.linalg.inv(np.dot(x_train.T,x_train)).dot(x_train.T).dot(y_train)
        self.coef_ = betas[1:]
        self.intercept_ = betas[0]
            
    def predict(self,x_test):
        return (np.dot(x_test,self.coef_) + self.intercept_)