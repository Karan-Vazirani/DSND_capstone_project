# Stock Price Prediction 
## Introduction
Stock market prediction is the act of trying to determine the future value of a company stock
or other financial instrument traded on an exchange. Here, my aim is to predict the next day's closing price of Tata Consultancy Pvt. Ltd stock.

## Libraries
* Numpy
* Pandas
* Scikit Learn
* Keras
* Matplotlib

## File Description
* <strong>Data Folder</strong>  
  It contains the historical data of Tata Consultancy Pvt. Ltd. stocks
* <strong>Model Folder</strong>  
  It contains the saved keras models in .h5 and .json format
* <strong>Result Folder</strong>  
  It contains all the plots, Score, and Prediction value in excel format
* <strong>Utils Folder</strong>  
  It contains all utility python scripts for plotting, training model, saving and loading models and creating input data
* <strong>prediction_single.py</strong>  
  This script helps in predicting next day's closing price using only one feature (opening price)
* <strong>hyperparameters.py</strong>
  This script helps in finding the best pair of hyperparameters that could help in refining the model
* <strong>feature_importance.py</strong>  
  This script helps in finding the best selecting of features for predicting the next day's closing price
* <strong>final_model.py</strong>  
  This script is the most optimized one for predicting the next day's closing price
  
## Results
Initially, the r2 score on testing data was 0.74709 and mse score was 28572.16 for predicting the next day's closing price.  

<img src='https://github.com/rajatsharma369007/DSND_capstone_project/blob/master/result/single_attribute/output_test.png' width=400px>

After hypermeter optimization the r2 score increased to 0.97720 and mse value decreased to 2575.15.  

<img src='https://github.com/rajatsharma369007/DSND_capstone_project/blob/master/result/hyperParaModels/output.png' width=400px>

Then, we applied feature importance after that the r2 score increased to 0.98379 and mse value decreased to 1830.83  

<img src='https://github.com/rajatsharma369007/DSND_capstone_project/blob/master/result/final_model/output_test.png' width=400px>

## Acknowledgement
Thanks to Yahoo Finance for providing Tata Consultancy historical data   
Thanks to https://colah.github.io/posts/2015-08-Understanding-LSTMs/  
Thanks to Machine Learning for Trading course by Udacity  
  
  
