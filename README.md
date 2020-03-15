# Regression Methods From Scratch
## SARCOS Data Set
All models with be ran using the SARCOS data set and there performance assessed. SARCOS is an inverse dynamics problem for a seven degrees-of-freedom SARCOS anthropomorphic robot arm. The data set contains 21 input dimensions and one output dimension.
[http://www.gaussianprocess.org/gpml/data/](http://www.gaussianprocess.org/gpml/data/)

## Toy Problem
In order to validate the regression models a toy problem was created.  The toy problem was designed to be a simple noise free data set so that the results can be easily predicted. It was designed to be non-linear and two dimensional in order to test the capability of all the models. The following equation was used to generate 500 data points for the toy data set.

![z = 0.5sin(x) + tanh(3y)](https://render.githubusercontent.com/render/math?math=z%20%3D%200.5sin(x)%20%2B%20tanh(3y))

Each time any data was used, it was shuffled and then split in to test and train data set with an 80:20 split respectively. A short python code was written in order to automate and make this process. This code also contains a feature scaling option which allows for the normalisation and standardisation of the data.

### K-NN

The first step for validating the k-NN model was to find the optimum k values for the toy data. Firstly, the data was normalised between 0 and 1. Normalising each parameter is important for Euclidean distance, as for each parameter to have equal importance the ranges must be identical. A range of k values from 2 to 40 was tested and the root mean square error (RMSE) and mean absolute error (MAE) of each k value was calculated. For this process multi-threading was utilised using the Joblib Python module. Multi-threading allows for different k values to run simultaneously, reducing the overall compute time. The optimum k was found to be 13. The low MAE and RSME values of 0.0213 and 0.0262 show that k-NN algorithm functions as intended.

![k Value Comparison](https://imgur.com/dxZYnOl.jpg)


### Linear Regression

### Random Forest
