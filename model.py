import numpy as np
import pandas as pd
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('zomata_new.csv')


print(df.head())

df.drop('Unnamed: 0',axis=1,inplace=True)
X=df.drop('rating',axis=1)
y=df['rating']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=2021)
#modlling
from lightgbm import LGBMRegressor
lgb=LGBMRegressor(random_state=2021)

## fitting
lgb.fit(X_train,y_train)

##predicting X_Test
y_pred=lgb.predict(X_test)

## Saving model
import pickle
pickle.dump(lgb,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

print(y_pred)
