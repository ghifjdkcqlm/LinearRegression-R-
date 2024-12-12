# import numpy as np
# class linear_regression: 
#     def __init__(self,lr = 0.001, k =1000):
#         self.lr = lr   #lr for learning rate
#         self.k = k  #k for no. of iterations
#         self.w = None #weights
#         self.b = None  #b


    

#     def training_fn(self, x , y): #(x,y) training sets

#         #no. of features/samples = x.shape

#         sample_no, feature_no = x.shape 

#         #we intialise our weight and b = 0
        

#         self.weights = np.zeros(feature_no)
#         self.b = 0 
#         #b is always just one value it is not a vector


#         #we wish to run the following model as the number of iterations set initially

        

#         for i in range(self.k):

#             #predicting by eqn y(cap) = (summation (w.x))+b

#             y_cap = np.dot(x,self.weights)+self.b


#             '''
#         dJ/dw = 1/N (summation 2x(y_cap-y))
#         dJ/db = 1/N (summation 2(y_cap-y))
#             '''


#             dw = (1/sample_no)*np.dot(x.T,(y_cap-y))
#             db = (1/sample_no)*np.sum(y_cap-y)


#         #upgradation in w and b 
        
#             self.weights = self.weights - self.lr*dw
#             self.b = self.b - self.lr*db

#     def prediction(self, x):
        

#         y_cap = np.dot(x,self.weights)+self.b
#         return y_cap





import numpy as np

class linear_regression: 
    def __init__(self, lr=0.001, k=1000):
        self.lr = lr   # Learning rate
        self.k = k     # Number of iterations
        self.weights = None  # Weights
        self.b = None        # Bias

    def training_fn(self, x, y):
        sample_no, feature_no = x.shape 
        
        self.weights = np.zeros(feature_no)

    
        self.b = 0 

        for i in range(self.k):
            y_cap = np.dot(x, self.weights) + self.b
            
            dw = (1/sample_no) * np.dot(x.T, (y_cap - y))
            db = (1/sample_no) * np.sum(y_cap - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.b -= self.lr * db

    def prediction(self, x):
        y_cap = np.dot(x, self.weights.reshape(-1,1)) + self.b
        return y_cap