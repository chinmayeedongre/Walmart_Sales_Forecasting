import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from datetime import date
from sklearn.metrics import mean_absolute_error
import datetime as dt



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet



stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')


train_df = train.sort_values('Date')

test_df = test.sort_values('Date')



train_df = train.merge(stores, how='left').merge(features, how='left')
print(train_df.shape)
#print(train_df.head(2))
print("               ")
print(train_df.columns)

test_df = test.merge(stores, how='left').merge(features, how='left')
print(test_df.shape)
#print(train_df.head(2))
print("               ")
print(test_df.columns)

train_df['Date'] = pd.to_datetime(train_df['Date'], format='%d-%m-%Y')
test_df['Date'] = pd.to_datetime(test_df['Date'], format='%Y-%m-%d')

train_df['Type'] = train_df['Type'].astype('category')
test_df['Type'] = test_df['Type'].astype('category')

train_df['IsHoliday'] = train_df['IsHoliday'].apply(lambda x: 0 if x == False else 1).astype('int64')
test_df['IsHoliday'] = test_df['IsHoliday'].apply(lambda x: 0 if x == False else 1).astype('int64')

train_df["Day"]= pd.DatetimeIndex(train_df['Date']).day
train_df['Month'] = pd.DatetimeIndex(train_df['Date']).month
train_df['Year'] = pd.DatetimeIndex(train_df['Date']).year

test_df["Day"]= pd.DatetimeIndex(test_df['Date']).day
test_df['Month'] = pd.DatetimeIndex(test_df['Date']).month
test_df['Year'] = pd.DatetimeIndex(test_df['Date']).year

train_df['MarkDown1'] = train_df['MarkDown1'].fillna(0)
train_df['MarkDown2'] = train_df['MarkDown2'].fillna(0)
train_df['MarkDown3'] = train_df['MarkDown3'].fillna(0)
train_df['MarkDown4'] = train_df['MarkDown4'].fillna(0)
train_df['MarkDown5'] = train_df['MarkDown5'].fillna(0)


test_df['MarkDown1'] = test_df['MarkDown1'].fillna(0)
test_df['MarkDown2'] = test_df['MarkDown2'].fillna(0)
test_df['MarkDown3'] = test_df['MarkDown3'].fillna(0)
test_df['MarkDown4'] = test_df['MarkDown4'].fillna(0)
test_df['MarkDown5'] = test_df['MarkDown5'].fillna(0)



test_df['Temperature'] = test_df['Temperature'].fillna(0)

test_df['Fuel_Price'] = test_df['Fuel_Price'].fillna(0)

test_df['Unemployment'] = test_df['Unemployment'].fillna(0)

test_df['CPI'] = test_df['CPI'].fillna(0)

seasons_dict = {
    1:"Winter",
    2:"Winter",
    3:"Spring",
    4:"Spring",
    5:"Spring",
    6:"Summer",
    7:"Summer",
    8:"Summer",
    9:"Fall",
    10:"Fall",
    11:"Fall",
    12:"Winter"}

train_df['Season'] = train_df['Month'].map(seasons_dict).astype('category')
test_df['Season'] = test_df['Month'].map(seasons_dict).astype('category')

train_df['Quarter'] = train_df['Date'].dt.quarter

test_df['Quarter'] = test_df['Date'].dt.quarter

train_df['Week_Number'] = train_df['Date'].dt.week.astype('int')

test_df['Week_Number'] = test_df['Date'].dt.week.astype('int')

feature_drop=['Unemployment']
train_df = train_df.drop(feature_drop, axis=1)
test_df = test_df.drop(feature_drop, axis=1)

train_df.loc[train_df['Size'] <= 50000,'store_size']= 'small'
train_df.loc[(train_df['Size'] > 50000) & (train_df['Size'] < 145000),'store_size']= 'mid'
train_df.loc[train_df['Size'] >= 145000,'store_size']= 'big'

train_df['store_size'] = train_df['store_size'].astype('category')


test_df.loc[test_df['Size'] <= 50000,'store_size']= 'small'
test_df.loc[(test_df['Size'] > 50000) & (test_df['Size'] < 145000),'store_size']= 'mid'
test_df.loc[test_df['Size'] >= 145000,'store_size']= 'big'

test_df['store_size'] = test_df['store_size'].astype('category')

train_df.loc[train_df['Temperature'] <= 20,'Temp_range']= 'Cold'
train_df.loc[(train_df['Temperature'] > 20) & (train_df['Temperature'] < 70),'Temp_range']= 'Moderate'
train_df.loc[train_df['Temperature'] >= 70,'Temp_range']= 'Hot'

train_df['Temp_range'] = train_df['Temp_range'].astype('category')


test_df.loc[test_df['Temperature'] <= 20,'Temp_range']= 'Cold'
test_df.loc[(test_df['Temperature'] > 20) & (test_df['Temperature'] < 70),'Temp_range']= 'Moderate'
test_df.loc[test_df['Temperature'] >= 70,'Temp_range']= 'Hot'

test_df['Temp_range'] = test_df['Temp_range'].astype('category')

train_df = pd.get_dummies(train_df, columns=['Type'])


test_df = pd.get_dummies(test_df, columns=['Type'])



train_df.loc[(train_df.Year==2010) & (train_df.Week_Number==13), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2011) & (train_df.Week_Number==16), 'IsHoliday'] = True
train_df.loc[(train_df.Year==2012) & (train_df.Week_Number==14), 'IsHoliday'] = True
test_df.loc[(test_df.Year==2013) & (test_df.Week_Number==13), 'IsHoliday'] = True


train_df['MarkDown'] = train_df['MarkDown1'] + train_df['MarkDown2'] + train_df['MarkDown3'] + train_df['MarkDown4'] + train_df['MarkDown5']

test_df['MarkDown'] = test_df['MarkDown1'] + test_df['MarkDown2'] + test_df['MarkDown3'] + test_df['MarkDown4'] + test_df['MarkDown5']



train_df1 =train_df
y = train_df1['Weekly_Sales']

X = train_df1.drop(['Weekly_Sales','Date','Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI','Temp_range','MarkDown','Season','store_size'],axis=1)

#train_df1 =train_df

X_query_merge = test_df.drop(['Date','Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI','Temp_range','MarkDown','Season','store_size'],axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.40)

X_train_merge = X_train.astype('int64')

X_test_merge =X_valid.astype('int64')

X_query_merge = X_query_merge.astype('int64')

clf_rf = RandomForestRegressor(max_depth=30,min_samples_split= 5,n_estimators=150,min_samples_leaf=1)

clf_rf.fit(X_train_merge,y_train)

#clf_et = ExtraTreesRegressor(max_depth=155,min_samples_leaf=1,min_samples_split=4,n_estimators=220)

#clf_et.fit(X_train_merge,y_train)

#clf_xg = XGBRegressor(learning_rate=0.1,n_estimators=735,max_depth=9,min_child_weight=1 )

#clf_xg.fit(X_train_merge,y_train)

import pickle
pickle.dump(clf_rf, open('model.pkl','wb'))
#pickle.dump(clf_et, open('model.pkl','wb'))
#pickle.dump(clf_xg, open('model.pkl','wb'))




model = pickle.load(open('model.pkl','rb'))

y_pred_train = model.predict(X_train_merge)
y_pred_test = model.predict(X_test_merge)
y_pred_query = model.predict(X_query_merge)

print(model.predict(X_train_merge))
print(model.predict(X_test_merge))
print(model.predict(X_query_merge))

print(mean_absolute_error(y_train,y_pred_train))
print(mean_absolute_error(y_valid,y_pred_test))

print(model.predict([[44,12,0,20000,23,2,2012,1,7,1,0,0]]))