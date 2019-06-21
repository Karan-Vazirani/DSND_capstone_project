'''
this script helps to predict closing price of dataset using neural network using
single attribute. Input to the model is the open price trend and the output of the 
model is the closing price of the stock.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import train
from utils import save_load
from utils import plot
from utils import tensor

# loading dataset
df = pd.read_csv('./data/TCS.NS.csv')
df = df.dropna()

split_ratio = 0.8
df_train = df[:int(len(df)*split_ratio)]
df_test = df[int(len(df)*split_ratio):]

# taking input set using open price and close price column
columns = [1, 4]

# important parameters
timestep = 60
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

X_train, y_train = tensor.create_tensor(training_set_scaled, timestep, input_col,\
                                        output_col, no_of_feature)

# creating testing set
testing_set = df_test.iloc[:, columns].values

# concatenating additional 60 days data from training set
x1 = pd.DataFrame(training_set[len(training_set)-timestep:])
x2 = pd.DataFrame(testing_set)
testing_set = np.array(pd.concat([x1, x2]))
testing_set_scaled = sc_input.transform(testing_set)

X_test, y_test = tensor.create_tensor(testing_set_scaled, timestep, input_col, \
                                      output_col, no_of_feature)

###############################################################################

epochs = 90
model = train.training(X_train, y_train, no_of_feature, epochs)

path_name = "./model/single_attribute"

# Saving the model
save_load.save_model(path_name, model)

###############################################################################

# loading the model
path_name = "./model/single_attribute"
model = save_load.load_model(path_name)

# prediction using train data
pred_train_scaled = model.predict(X_train)

# rescaling the predictions (train data)
train_predict = sc_output.inverse_transform(pred_train_scaled)
train_actual = sc_output.inverse_transform(y_train) 

print('R2 Score : ', r2_score(train_actual, train_predict))
print('MSE Score : ', mean_squared_error(train_actual, train_predict))

plot.time_series_plot(train_actual, train_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', \
                 'Neural Network (single attribute - train data)')


# prediction using test data
pred_test_scaled = model.predict(X_test)

# rescaling for predictions (test data)
test_predict = sc_output.inverse_transform(pred_test_scaled)
test_actual = sc_output.inverse_transform(y_test)

print('R2 Score : ', r2_score(test_actual, test_predict))
print('MSE Score : ', mean_squared_error(test_actual, test_predict))

plot.time_series_plot(test_actual, test_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', \
                 'Neural Network (single attribute - test data)')

###############################################################################

# saving the results in excel format
date = pd.DataFrame(df_test['Date']).reset_index(drop=True)
actual_price_df = pd.DataFrame(test_actual).round(3)
predict_price_df = pd.DataFrame(test_predict).round(3)
combined_df = pd.concat([date, actual_price_df, predict_price_df], axis = 1 )
combined_df.columns = ['date','actual_close', 'predicted_close']
combined_df.to_excel('./result/single_attribute/result.xlsx', index = False)

# R2 Score :  0.7470900859501621 MSE Score :  28572.169402722284
# R2 Score :  0.9951565494031501 MSE Score :  568.1676091507665