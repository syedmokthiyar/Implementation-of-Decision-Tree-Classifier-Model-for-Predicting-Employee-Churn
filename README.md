# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. From sklearn import LabelEncoder to convert string values to numerical values.
4. Assign the train dataset and test dataset.
5. From sklearn.tree import DecisionTreeClassifier.
6. From sklearn import metrics.
7. Find the accuracy of our model and predict the require values. 

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Syed Mokthiyar S.M
RegisterNumber:  212222230156
```
```python
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```


## Output:
![Screenshot 2024-04-02 161642](https://github.com/syedmokthiyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787294/5080869e-abbe-4fee-890d-e1084b8eb802)


# Accuracy
![Screenshot 2024-04-02 161550](https://github.com/syedmokthiyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787294/64074082-f9ef-459f-8051-de6953ed0cdc)

# dt.predict
![Screenshot 2024-04-02 155625](https://github.com/syedmokthiyar/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787294/c9e7f9e5-0153-4941-b72f-4138123858a3)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
