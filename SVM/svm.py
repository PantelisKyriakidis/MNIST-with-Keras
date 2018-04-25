from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#import dataset
# the data, shuffled and split between train and test sets
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#                         PREPROCESSING DATA

# Change from matrix to array --> dimension 28x28 to array of dimention 784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Change to float datatype
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Scale the data to lie between -1 to 1
x_train = x_train / 255.0*100 - 50
x_test = x_test / 255.0*100 - 50
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#I mod2 the label sets to have a result of 1 or 0 , for odds and evens respectively
y_train =y_train % 2
y_test =y_test % 2

# PCA
pca = PCA(n_components=50)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#                         GRID SEARCH FOR PARAMETER OPTIMIZING

svm = SVC()
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
                    #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
print("grid search")
grid = GridSearchCV(svm, parameters, verbose=3)
print("grid.fit")
grid.fit(x_train[0:7000], y_train[0:7000]) #grid search learning the best parameters
print("grid done")

print (grid.best_params_)

#Now we train the best estimator in the full dataset
print("training svm")
best_svm = grid.best_estimator_
best_svm.fit(x_train , y_train)
print("svm done")


print("Testing")
print("score: ", best_svm.score(x_test, y_test,))