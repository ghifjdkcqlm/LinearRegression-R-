# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib 
# import seaborn as sns
# import lr_my_files 

# rr_df = pd.read_csv(r'C:\Users\dimri\OneDrive\Desktop\vs code java\myfirstvscode\regression task\robosheet1.csv')

# rr_df.drop(rr_df[rr_df.AGE<1].index, inplace = True)

# x = rr_df[['AGE', 'FEMALE', 'LOS', 'RACE', 'APRDRG']].values
# y = rr_df['TOTCHG'].values
# print(x.shape, y.shape)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV dataset

data = pd.read_csv(r'C:\Users\dimri\OneDrive\Desktop\vs code java\myfirstvscode\regression task\robosheet1.csv')

data = data.dropna(subset=['RACE'])

# Separate features (X) and target variable (y)
X = data[['AGE', 'LOS', 'FEMALE', 'APRDRG', 'RACE']].values  # Select relevant features and convert to NumPy array
y = data['TOTCHG'].values  # Target variable

# Data preprocessing (if needed)
# Handle missing values, encode categorical variables, etc.
X = X[~np.isnan(X).any(axis=1)]
y = y[~np.isnan(y)]

data.drop(data[data.AGE<1].index, inplace = True)


# Define your linear regression class
class linear_regression: 
    def __init__(self, lr=0.0001, k=10000):
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

# Initialize and train your linear regression model
model = linear_regression(lr=0.001, k=1000)
model.training_fn(X, y)

# Get the coefficients (weights) and bias term
coefficients = model.weights
bias = model.b



#the coeffecient part
# Display coefficients and bias term
print("Coefficients:", coefficients)
print("Bias:", bias)





#the graph part
# Plotting the actual vs. predicted values
plt.scatter(y, model.prediction(X).flatten())
plt.xlabel('Actual TOTCHG')
plt.ylabel('Predicted TOTCHG')
plt.title('Actual vs. Predicted TOTCHG')
plt.show()