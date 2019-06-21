'''
this script helps to select the best features which contribute the most in the
prediction.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import train
from utils import save_load
from utils import tensor
from utils import plot
import os

# loading dataset
df = pd.read_csv('./data/TCS.NS.csv')
df = df.dropna()
df = pd.concat([df, pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'\
                                 ])], axis=1)

split_ratio = 0.8
df_train = df[:int(len(df)*split_ratio)]
df_test = df[int(len(df)*split_ratio):]

# important parameters
columns = [1, 4, 6, 7]
no_of_feature = 4
timestep = 60
input_col = [0, 1, 2, 3]
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

X_train, y_train = tensor.create_tensor(training_set_scaled, timestep, input_col,\
                                        output_col, no_of_feature)

# creating testing set
testing_set = df_test.iloc[:, columns].values
x1 = pd.DataFrame(training_set[len(training_set)-timestep:])
x2 = pd.DataFrame(testing_set)
testing_set = np.array(pd.concat([x1, x2]))
testing_set_scaled = sc_input.transform(testing_set)

X_test, y_test = tensor.create_tensor(testing_set_scaled, timestep, input_col, \
                                      output_col, no_of_feature)
# ============================================================================

count = 0
epochs = 90

combination = []

# creating pairs of feature attributes
from itertools import combinations
for i in range(1,len(input_col)+1):
    combination.append(list((combinations(input_col, i))))


for i in range(no_of_feature):
    for j in range(len(combination[i])):
        feature = np.array(combination[i][j])
        model = train.training(X_train[:, :, feature], y_train, feature.shape[0]\
                               , epochs, 'relu', 'adam')
        # Saving the model        
        path_name = "./model/feature_importance" + "/" + str(count)
        os.mkdir(path_name)
        save_load.save_model(path_name, model)
        count = count + 1

# =============================================================================

path_name = "./model/feature_importance" 

# actual output
test_actual = sc_output.inverse_transform(y_test)

# creating dataframe for storing result
results = pd.DataFrame(columns=['feature_col', 'r2_score', 'mse_score'])

count = 0
for i in range(no_of_feature):
    for j in range(len(combination[i])):
        feature = np.array(combination[i][j])
        # loading the model
        model = save_load.load_model(path_name + "/" + str(count))
        # prediction
        pred_test_scaled = model.predict(X_test[:, :, feature])
        test_predict = sc_output.inverse_transform(pred_test_scaled)
        # evaluation
        model_accuracy_r2 = r2_score(test_actual, test_predict)
        model_accuracy_mse = mean_squared_error(test_actual, test_predict)
        print("feature: {}\n r2_score: {}\n mse_score: {}\n".format(feature, \
              model_accuracy_r2, model_accuracy_mse))
        plot.time_series_plot(test_actual, test_predict, 'red', 'blue', \
                              'actual_close', 'predicted_close', 'days', 'price',\
                              'Neural Network (multiple attributes - train data)')

        results.loc[count] = [feature, model_accuracy_r2, model_accuracy_mse]
        count = count + 1

results.to_excel("./result/feature_importance/result.xlsx")

# r2_score: 0.9837941366548775 mse_score: 1830.8363851761196