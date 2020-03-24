from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#--------------------------------------------------
#DataPreprocessing
#--------------------------------------------------
#Loading the iris dataset
iris = datasets.load_iris()
#Getting data and labels
#Data
X = iris.data
#Labels
y = iris.target #labels
#Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
iterations=400
learning_rate=0.1

ppn = Perceptron(max_iter=iterations, eta0=learning_rate, random_state=0)
#Train the perceptron
ppn.fit(X_train_std, y_train)

#--------------------------------------------------
#Test
#--------------------------------------------------
y_predicted = ppn.predict(X_test_std)
print("Predicted: ", y_predicted)
print("Real: ", y_test)

#--------------------------------------------------
#Metrics
#--------------------------------------------------
accuracy = accuracy_score(y_test, y_predicted)
print("Accuracy =  ", accuracy)

print("")