import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df= pd.read_csv('mail_data.csv')

data= df.where((pd.notnull(df)),'')

data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1

x=data['Message']
y=data['Category']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)

feature_extraction= TfidfVectorizer(min_df=1,stop_words='english', lowercase=True)
x_train_feature=feature_extraction.fit_transform(x_train)
x_test_feature=feature_extraction.transform(x_test)

y_train=y_train.astype('int')
y_test=y_test.astype('int')

model=LogisticRegression()
model.fit(x_train_feature,y_train)

prediction_on_training_data=model.predict(x_train_feature)
accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)

print("accuracy on training data :",accuracy_on_training_data)

prediction_on_testing_data=model.predict(x_test_feature)
accuracy_on_testing_data=accuracy_score(y_test,prediction_on_testing_data)

print("accuracy on testing data :",accuracy_on_testing_data)

content = input("enter mail content: ")
input_data_features=feature_extraction.transform([content])
prediction=model.predict(input_data_features)
print(prediction)
if prediction[0]==1:
    print("Ham mail")
else:
    print("Spam mail")