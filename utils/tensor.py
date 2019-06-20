'''
This script helps is creating the 3d input tensors
'''

# importing libraries
import numpy as np

# function for creating tensor
def create_tensor(df, timestep, input_col, output_col, no_of_feature):
    X = []
    y = []
    for i in range(timestep, len(df)):
        X.append(df[i-timestep: i, input_col])
        y.append(df[i, output_col])        
    
    # converting to numpy array
    X, y = np.array(X), np.array(y)
    # Reshaping to create 3D tensor
    X = X.reshape(X.shape[0], X.shape[1], no_of_feature)
    
    return X, y

