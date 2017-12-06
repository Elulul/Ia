# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:50:59 2017

@author: lulu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import keras
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split



df = pd.read_csv("D:/Telecom Nancy/IA/Projet/communities.data.csv", sep=";",decimal=".")
        


def CountMissingValue(df):
    res = np.arange(df.shape[1])
    for i in range(1,(df.shape[1])):
        cpt = 0
        for j in range(1,df.shape[0]):
            if(str(df.iloc[j,i]) == '?'):
                cpt = cpt + 1
        res[i] = cpt
    return res

def CountMissingValueNan(df):
    res = np.arange(df.shape[1])
    for i in range(1,(df.shape[1])):
        cpt = 0
        for j in range(1,df.shape[0]):
            if(np.isnan(df.iloc[j,i])):
                cpt = cpt + 1
        res[i] = cpt
    return res

def replaceMissingValue(df2,val):
    for i in range(1,df.shape[1]):
        for j in range(1,df.shape[0]):
            if(str(df.iloc[j,i]) == '?'):
                df2.iloc[j,i] = (df2.iloc[:,[val]]).mean[0]


res = CountMissingValue(df)
df2 = df



newdf = df.replace('?',np.nan)
res = newdf.apply(lambda x: sum(x.isnull().values), axis = 0)

ColumnToSuppress = []

for i in range(1,res.size):
   
    if (res[i] > df.shape[0]/2):
        ColumnToSuppress.append(i)
       
for i in range(1,res.size):
   
    if (res[i] > df.shape[0]/2):
        ColumnToSuppress.append(i)
       

#replaceMissingValue(df2,28)






df2 = newdf.drop(newdf.columns[ColumnToSuppress],axis = 1)

df2 = df2.drop('communityname',axis = 1)

#df2['OtherPerCap'] = df2['OtherPerCap'].fillna((df2['OtherPerCap'].mean()))
#m = df2.corr()

df2['OtherPerCap'].fillna(df2.iloc[:,[28]].mean()[0], inplace=True)


res = df2.apply(lambda x: sum(x.isnull().values), axis = 0)
m = df2.corr()
hetMap=sns.heatmap(df2.corr())

corr = []


for i in range(1,m.shape[0]):
    for j in range(1,m.shape[1]):
        if(m.iloc[i,j] > 0.9 or m.iloc[i,j] < -0.9):
            if(m.columns[i] != m.columns[j] ):
                l = [] 
                l.append(m.columns[i])
                l.append(m.columns[j])
                corr.append(l)
        
        
# A ameliorer faire une liste variable qu'il vaut mieux garder plutot que de supprimer arbitrairement        
def ListWithSuppressionOfCorrelatedVariable(corr):
    ToSuppress = []
    i = 0
    while( i < len(corr)):
        if not(corr[i][0] in ToSuppress):
            tampon = corr[i][0]
            ToSuppress.append(corr[i][0])
            while (corr[i][0]==tampon):
                i = i +1
                if not(corr[i][1] in ToSuppress):
                    ToSuppress.append(corr[i][1])
        i = i + 1  
    return(ToSuppress)
        
ToSuppress = ListWithSuppressionOfCorrelatedVariable(corr)


def SuppressionOfCorrelatedVariable(ToSuppress,df):
    dfres = df
    for item in ToSuppress:
        print(item)
        dfres = dfres.drop(item,axis = 1)
    return dfres

df3 = SuppressionOfCorrelatedVariable(ToSuppress,df2)
    
df2.shape[0]



train, validate, test = np.split(df3.sample(frac=1), [int(.6*len(df3)), int(.8*len(df3))])

y = df3[df3.columns[df3.shape[1]-1]]
x = df3.drop(df3.columns[df3.shape[1]-1],axis = 1)

y2 = df2[df2.columns[102]]
x2 = df2.drop(df2.columns[102],axis = 1)


xtrain, xtest,ytrain, ytest = train_test_split( x, y, test_size=0.33, random_state=42)

xtrain2, xtest2,ytrain2, ytest2 = train_test_split( x2, y2, test_size=0.33, random_state=42)




df3.shape[1]


#y = df2[df2.columns[df2.shape[1]-1]]
#x = df2.drop(df2.columns[df2.shape[1]-1],axis = 1)
#
#xtrain, xtest,ytrain, ytest = train_test_split( x, y, test_size=0.33, random_state=42)


#ytrain = train.values[:,38 ]
#xtrain = train.drop(train.columns[38],axis = 1)
#xtrain = xtrain.values
#
#yvalidate = validate.values[:,38 ]
#xvalidate = validate.drop(validate.columns[38],axis = 1)
#xvalidate = xvalidate.values
#
#ytest = test[test.columns[38]]
#xtest = test.drop(test.columns[38],axis = 1)
#xtest = xtest.values


sc = StandardScaler()
xtrain2 = sc.fit_transform(xtrain2)
xtest2 = sc.transform(xtest2)

xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)



#model = Sequential()
##model.add(Dense(30, input_dim=102, kernel_initializer='normal', activation='relu'))
#model.add(Dense(51, input_dim=102, kernel_initializer='normal', activation='relu'))
#
#model.add(Dense(51, kernel_initializer='normal', activation='relu'))
##model.add(Dense(30, kernel_initializer='normal', activation='relu'))
##model.add(Dense(30, kernel_initializer='normal', activation='relu'))
#
#model.add(Dense(1, kernel_initializer='normal'))
#
#
#
#
#
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(xtrain2,ytrain2,batch_size = 5,epochs = 100)
#   
#    
#ypred = model.predict(xtest2)    
#    
#r2_RN_all_var = r2_score(ytest2,ypred)
#mse_RN_all_var = mean_squared_error(ytest2,ypred)
#
#
#
#
#model = Sequential()
##model.add(Dense(30, input_dim=102, kernel_initializer='normal', activation='relu'))
#model.add(Dense(df3.shape[1] // 2, input_dim=df3.shape[1] - 1, kernel_initializer='normal', activation='relu'))
#
#model.add(Dense(df3.shape[1] // 2, kernel_initializer='normal', activation='relu'))
##model.add(Dense(30, kernel_initializer='normal', activation='relu'))
##model.add(Dense(30, kernel_initializer='normal', activation='relu'))
#
#model.add(Dense(1, kernel_initializer='normal'))
#
#
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(xtrain,ytrain,batch_size = 5,epochs = 100)
#
#ypred = model.predict(xtest)    
#    
#r2_RN = r2_score( ytest,ypred)
#mse_RN = mean_squared_error(ytest,ypred)
#


from sklearn.tree import DecisionTreeRegressor
treeRegressor = DecisionTreeRegressor(criterion = "friedman_mse",random_state = 0)
treeRegressor = DecisionTreeRegressor(criterion = "mae",random_state = 0)
treeRegressor = DecisionTreeRegressor(criterion = "mse",random_state = 0)


treeRegressor.fit(xtrain2,ytrain2)

ypred2 = treeRegressor.predict(xtest2)

mse_tr_all_var = mean_squared_error(ypred2,ytest2)
r2_tr_all_var = r2_score(ytest2,ypred2)

best_var  = treeRegressor.feature_importances_
var_data = pd.DataFrame(data = column_stack([df2.drop('ViolentCrimesPerPop',axis = 1).columns,best_var]))
var_data.sort_values(1)



treeRegressor.fit(xtrain,ytrain)

ypred = treeRegressor.predict(xtest)

mse_tr = mean_squared_error(ypred,ytest)
r2_tr = r2_score(ytest,ypred)



from sklearn.ensemble import RandomForestRegressor
regressor =  RandomForestRegressor(n_estimators = 200,max_features = "log2")  
regressor.fit(xtrain,ytrain) 

ypred = regressor.predict(xtest)

mse_rdf = mean_squared_error(ypred,ytest)
r2_rdf = r2_score(ytest,ypred)



regressor.fit(xtrain2,ytrain2) 
ypred = regressor.predict(xtest2)

mse_rdf_all_var = mean_squared_error(ypred,ytest2)
r2_rdf_all_var = r2_score(ytest2,ypred)







