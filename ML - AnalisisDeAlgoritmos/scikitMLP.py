#from sklearn import datasets
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#--------------------------------------------------
#DataPreprocessing
#--------------------------------------------------
#Loading the iris dataset
#inData = datasets.load_iris()
my_data = genfromtxt('TSLA.csv', delimiter=',')
filas, columnas = my_data.shape
days = np.zeros(filas-1)
values = np.zeros((filas-1,columnas-1))    
for i in range(0,filas-1):
    days[i] = i
    for j in range(0,columnas-1):
        values[i,j] = my_data[i+1,j+1]

#Getting data and labels
#Data
X = values
#Labels
y = days
#Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False)
#Standarizing all the featires to have zero mean and unit variance
#Training the standard scaler
sc= StandardScaler()
sc.fit(X_train)
#Aply the scaler to the X training and test data
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#--------------------------------------------------
#Train a Perceptron Learner
#--------------------------------------------------
#Creating a perceptron object with the following parameters:
iterations=10
l_r=1

mlp = MLPClassifier(hidden_layer_sizes = (10,1000), max_iter=iterations, learning_rate='adaptive', random_state=0)
#Train the perceptron
mlp.fit(X_train_std, y_train)

#--------------------------------------------------
#Test
#--------------------------------------------------
y_predicted = mlp.predict(X_test_std)
#print("Predicted: ", y_predicted)
#print("Real: ", y_test)

#ZeroArray = np.zeros(725)
#--------------------------------------------------
#Metrics
#--------------------------------------------------
MSE = mean_squared_error(y_test, y_predicted)
#MSE = mean_squared_error(y_test, ZeroArray)
Error = (MSE * 100) / 4258611.0

print("MSE =  ", Error)

print("")