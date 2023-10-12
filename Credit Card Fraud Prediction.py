# import required modules
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# loading dataset
credit_card_dataset = pd.read_csv("creditcard.csv")
# show the first and last rows in the dataset
credit_card_dataset.head()
credit_card_dataset.tail()
# show the dataset shape
credit_card_dataset.shape
# show some statistical info about the dataset
credit_card_dataset.describe()

# check if there is any none values in the dataset
credit_card_dataset.isnull().sum()
# make data cleaning
credit_card_dataset= credit_card_dataset.dropna()


# show the groups in the output and its repetition and plot it
print(credit_card_dataset['Class'].value_counts())
sns.catplot(x = 'Class',data=credit_card_dataset,kind='count')
plt.show()
# show the distribution of the output feature
sns.distplot(credit_card_dataset['Class'],color='red')
plt.show()



# The data is unbalanced so will divide it into legitimate and fraud datasets to balance two groups
legitimate = credit_card_dataset[credit_card_dataset['Class']==0]
print(credit_card_dataset.shape,legitimate.shape)
legitimate['Amount'].describe()
fraud = credit_card_dataset[credit_card_dataset['Class']==1]
print(credit_card_dataset.shape,fraud.shape)
fraud['Amount'].describe()
# the size of fraud dataset is 204
# balance the data by take 204 twos randomly from legitimate dataset
legitimate = legitimate.sample(n=204)
print(credit_card_dataset.shape,legitimate.shape)

# concatinating two dataset in original dataset credit_card_dataset
new_credit_card_dataset = pd.concat([legitimate,fraud],axis=0)
new_credit_card_dataset.head()
new_credit_card_dataset.tail()
new_credit_card_dataset['Class'].value_counts()
new_credit_card_dataset.groupby('Class').mean()
# find the correlation between all features in the dataset
correlation_values = new_credit_card_dataset.corr()
# plot the correlation between the features
plt.figure(figsize=(15,15))
sns.heatmap(correlation_values,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')




# Spliting the data into input and label data
X = new_credit_card_dataset.drop(columns='Class',axis=1)
Y = new_credit_card_dataset['Class']
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)

# create LogisticRegression model and train it
LRModel = LogisticRegression()
LRModel.fit(x_train,y_train)
# make the model predict from train data
predicted_train_data = LRModel.predict(x_train)
# make the model predict from test data
predicted_test_data = LRModel.predict(x_test)
# find the accuray of the model on train data prediction
accuracy_train_data = accuracy_score(predicted_train_data,y_train)
# find the accuray of the model on test data prediction
accuracy_test_data = accuracy_score(predicted_test_data,y_test)
print(accuracy_train_data,accuracy_test_data)



# making a predictive system 
input_data = (125796,-0.0163229823233489,0.198164839726069,-0.536648508581257,-1.96210424089796,0.0524176830347094,-0.0996721121126959,0.150980374189302,0.381031022665012,-1.52477600242462,-0.0282946462611107,0.384120654536603,0.361353855460271,0.708091991806141,0.0722449702844524,-1.64291448664949,1.01182039323882,0.0941578703896646,-1.26071934493168,1.02711132959967,0.0403315166363104,0.133618066877627,0.151264700524407,0.123349791688918,0.298586411634562,-0.198451332911796,-0.425973342945227,-0.132138580255892,-0.0743934201432191,52)
# convert input data into 1D numpy array
input_array = np.array(input_data)
# convert 1D input array into 2D array
input_2D_array = input_array.reshape(1,-1)
# make the model predict from input data
if LRModel.predict(input_2D_array)[0]==1:
    print("this credit card is fraud")
else:
    print("this credit card is legit")



