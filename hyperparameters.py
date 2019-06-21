'''
this script is for hyperparameter optimisation where we will get best pair of
hyperparameters.
'''
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import save_load
from utils import plot
from utils import tensor
from utils import train
import os

# loading dataset
df = pd.read_csv('./data/TCS.NS.csv')
df = df.dropna()

split_ratio = 0.8
df_train = df[:int(len(df)*split_ratio)]
df_test = df[int(len(df)*split_ratio):]

# taking input set using open price and close price column
columns = [1, 4]

# important parameters
epochs = 90
no_of_feature = 1
input_col = [0]
output_col = [1]

# creating minmax scaler object
input_set = df.iloc[:, columns].values
sc_input = MinMaxScaler(feature_range = (0,1))
sc_input.fit(input_set)
sc_output = MinMaxScaler(feature_range = (0,1))
sc_output.fit(input_set[:,output_col])

# creating training set
training_set = df_train.iloc[:, columns].values
training_set_scaled = sc_input.transform(training_set)

# hyperparameters
neurons = [5, 60, 80] 
optimiser = ['adam', 'rmsprop']
activation = ['tanh', 'relu', 'sigmoid']

count = 0
# ============================================================================

for neuron in neurons:
    # creating 3d tensor
    X_train, y_train = tensor.create_tensor(training_set_scaled, neuron, \
                                            input_col, output_col, no_of_feature)
    for optim in optimiser:
        for func in activation:
            # fitting the model
            model = train.training(X_train, y_train, no_of_feature, epochs, \
                                   func, optim)
            
            # Saving the model
            path_name = "./model/hyperParaModels" + "/" + str(count)
            os.mkdir(path_name)
            save_load.save_model(path_name, model)
            count = count + 1
                
# =============================================================================

path_name = "./model/hyperParaModels" 

results = pd.DataFrame(columns=['neuron', 'optim', 'activation', 'r2_score', 'MSE'])

count=0

for neuron in neurons:
    # creating testing set
    testing_set = df_test.iloc[:, columns].values
    # concatenating additional 60 days data from training set
    x1 = pd.DataFrame(training_set[len(training_set)-neuron:])
    x2 = pd.DataFrame(testing_set)
    testing_set = np.array(pd.concat([x1, x2]))
    testing_set_scaled = sc_input.transform(testing_set)
    # creating 3d tensor
    X_test, y_test = tensor.create_tensor(testing_set_scaled, neuron, input_col,\
                                          output_col, no_of_feature)

    for optim in optimiser:
        for func in activation:
            # loading model
            model = save_load.load_model(path_name + "/" + str(count))
            # prediction using test data
            pred_test_scaled = model.predict(X_test)
            # rescaling the predictions
            test_predict = sc_output.inverse_transform(pred_test_scaled)
            test_actual = sc_output.inverse_transform(y_test)
            # score
            model_accuracy_r2 = r2_score(test_actual, test_predict)
            model_accuracy_mse = mean_squared_error(test_actual, test_predict)
            print("r2 : ", model_accuracy_r2)
            print("mse : ", model_accuracy_mse)
            
            plot.time_series_plot(test_actual, test_predict, 'red', 'blue', 'actual_close',\
                     'predicted_close', 'days', 'price', 'Neural Network')
            
            count = count +1
            
            results.loc[count] = [neuron, optim, func, model_accuracy_r2, \
                       model_accuracy_mse]

results.to_excel("./model/hyperParaModels/hyperparameter_optim.xlsx")

# r2 :  0.9772057086220985, mse :  2575.1554940470696