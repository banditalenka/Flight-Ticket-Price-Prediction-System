import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import json
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
reg_rf=RandomForestRegressor()

cv= ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
lr_clf=LinearRegression()


def find_best_model_using_gridsearchcv(X,Y):
    algos={
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        },

    }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(X,Y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

matplotlib.rcParams["figure.figsize"]=(20,10)
df1=pd.read_excel(r"Train_DataSet_New.xlsx")
pd.set_option('display.max_columns',None)
#print(df1)
#print(df1.info())
#print(df1.isnull().sum())
df2=df1.dropna()
df2["Date_Diff"]=(pd.to_datetime(df2.Date_of_Journey,format="%d/%m/%Y")-pd.to_datetime(df2.Date_of_Booking,format="%Y-%m-%d")).dt.days
df2["Journey_day"]=pd.to_datetime(df2.Date_of_Journey,format="%d/%m/%Y").dt.day
df2["Journey_month"]=pd.to_datetime(df2["Date_of_Journey"],format="%d/%m/%Y").dt.month
#print(df2)
df2.drop(["Date_of_Journey"],axis=1,inplace=True)
df2["Booking_day"]=pd.to_datetime(df2.Date_of_Booking,format="%d/%m/%Y").dt.day
df2["Booking_month"]=pd.to_datetime(df2["Date_of_Booking"],format="%d/%m/%Y").dt.month
df2.drop(["Date_of_Booking"],axis=1,inplace=True)
df2["Dep_hour"]=pd.to_datetime(df2["Dep_Time"]).dt.day
df2["Dep_min"]=pd.to_datetime(df2["Dep_Time"]).dt.minute
df2.drop(["Dep_Time"],axis=1,inplace=True)
df2.drop(["Arrival_Time"],axis=1,inplace=True)
df5=df2.copy()
#print(df5)
duration=list(df5["Duration"])
for i in range(len(duration)):
    if len(duration[i].split())!=2:
        if "h" in duration[i]:
            duration[i]=duration[i].strip()+" 0m"
        else:
            duration[i]="0h "+duration[i]

duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

df5["Duration_hours"]=duration_hours
df5["Duration_mins"]=duration_mins
df5.drop(["Duration"],axis=1,inplace=True)
df6=df5.copy()
#print(df6)
#print(df6["Airline"].unique())
#print(df6["Airline"].value_counts())
#print(pd.get_dummies(df6.Airline))
Airline=df6[["Airline"]]
Airline=pd.get_dummies(Airline,drop_first=True)
#print(df6)
#print(df6["Source"].value_counts())
Source=df6[["Source"]]
Source=pd.get_dummies(Source,drop_first=True)
Destination=df6[["Destination"]]
Destination=pd.get_dummies(Destination,drop_first=True)
#print(df6["Route"])
df6.drop(["Route","Additional_Info"],axis=1,inplace=True)
#print(df6["Total_Stops"].value_counts())
df6.replace({"non-stop":0,"1 stop": 1 ,"2 stops": 2 ,"3 stops": 3 ,"4 stops": 4},inplace=True)
df7=df6.copy()
#print(df7["Total_Stops"].value_counts())
#pd.set_option('display.max_columns',None)
#print(df7)
df8=pd.concat([df7,Airline,Source,Destination],axis=1)
df8.drop(["Airline","Source","Destination"],axis=1,inplace=True)
df9=df8.copy()
#print(df9.columns)
X=df9.drop('Price',axis='columns')
#print(X)
y=df9.Price
#print(y)
X_train, X_test, Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=10)
lr_clf.fit(X_train,Y_train)
#print(cross_val_score(LinearRegression(),X,y,cv=cv))
print(find_best_model_using_gridsearchcv(X,y))
reg_rf.fit(X_test,Y_test)
y_pred=reg_rf.predict(X_test)
print("4    RandomForest     "+str(reg_rf.score(X_train,Y_train)))
print("5    RandomForest     "+str(reg_rf.score(X_test,Y_test)))
#print(df9.info())
"""
with open('Flight_rf_diff.pickle','wb') as f:
    pickle.dump(reg_rf,f)


columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open("columns_diff.json","w") as f:
    f.write(json.dumps(columns))"""