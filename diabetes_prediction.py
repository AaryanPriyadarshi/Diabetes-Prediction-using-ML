
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')#Loading the dataset to the pandas dataframe

diabetes_dataset.head()#printing the first 5 rows of the dataset
diabetes_dataset.shape #number of rowns and columns in the dataset
diabetes_dataset.describe()#For getting stastistical measure of the data
diabetes_dataset.value_counts()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()

#Seperating datas and labels
x = diabetes_dataset.drop(columns= 'Outcome', axis= 1)
y = diabetes_dataset['Outcome']
print (x)
print (y)

scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print(standardized_data)
x = standardized_data
y = diabetes_dataset['Outcome']
print (x)
print (y)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, stratify= y, random_state= 2)
print(x.shape, x_train.shape, x_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)#training teh support vector machine classifier

#accuracy score on training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Acuracy score of the training data : ', training_data_accuracy)

#accuracy score on test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy score of the test data', test_data_accuracy)

input_data = (5,166,72,19,75,25.8,0.587,51)

input_data_as_np_array = np.asarray(input_data)#changing the input data to numpy array
input_data_reshaped = input_data_as_np_array.reshape(1,-1)#reshape the array as we are prediciting for one instance

std_data = scaler.transform(input_data_reshaped)#standardize the input data
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('This person is not Diabetic')
else:
    print('This person in Diabetic')
