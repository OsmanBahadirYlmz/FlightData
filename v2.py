import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn import tree
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder # Perform OneHotEnconding
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict # Cross Validation
from sklearn.linear_model import LogisticRegression # Modelling



# In[] data read
data=pd.read_csv('flightdata.csv')
df = pd.read_csv('flightdata.csv')

# In[] summirize visuilze
pd.DataFrame({'unicos':data.nunique(),
              'missing': data.isna().sum()/data.count(),
              'tipo':data.dtypes})

# In[] preprocessıng

#Saperating data
filtered_df = df[(df['ARR_DEL15'].isnull()) & ((df['CANCELLED'] != 0) & (df['DIVERTED'] != 0))]
df2=df[(df['ARR_DEL15'].isnull()) ]
df=df[df['ARR_DEL15'].notnull()]

print(df.columns)

dummycolumns=['YEAR','CRS_DEP_TIME','DEP_DELAY',
       'UNIQUE_CARRIER', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID',
       'DEST_AIRPORT_ID', 'CRS_ARR_TIME', 'ARR_DELAY',
        'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME','CANCELLED', 'DIVERTED',
        'Unnamed: 25']

for i in dummycolumns:
    try:        
        df = df.drop(i, axis=1)
    except:
        print('this column can not be deleted: ',i)
        continue
        






# In[] data preprocessing

#timeblk
# Helper function to create ARR_TIME_BLOCK
def arr_time(x):

  if x >= 600 and x <= 659:
    return '0600-0659'
  elif x>=1400 and x<=1459:
    return '1400-1459'
  elif x>=1200 and x<=1259:
    return '1200-1259'
  elif x>=1500 and x<=1559:
    return '1500-1559'
  elif x>=1900 and x<=1959:
    return '1900-1959'
  elif x>=900 and x<=959:
    return '0900-0959'
  elif x>=1000 and x<=1059:
    return  '1000-1059'
  elif x>=2000 and x<=2059:
    return '2000-2059'
  elif x>=1300 and x<=1359:
    return '1300-1359'
  elif x>=1100 and x<=1159:
    return '1100-1159'
  elif x>=800 and x<=859:
    return '0800-0859'
  elif x>=2200 and x<=2259:
    return '2200-2259'
  elif x>=1600 and x<=1659:
    return '1600-1659'
  elif x>=1700 and x<=1759:
    return '1700-1759'
  elif x>=2100 and x<=2159:
    return '2100-2159'
  elif x>=700 and x<=759:
    return '0700-0759'
  elif x>=1800 and x<=1859:
    return '1800-1859'
  elif x>=1 and x<=559:
    return '0001-0559'
  elif x>=2300 and x<=2400:
    return '2300-2400'

df['DEP_TIME'] = df['DEP_TIME'].astype('int')
df['DEP_TIME_BLOCK'] = df['DEP_TIME'].apply(lambda x :arr_time(x))

df['ARR_TIME'] = df['ARR_TIME'].astype('int')
df['ARR_TIME_BLOCK'] = df['ARR_TIME'].apply(lambda x :arr_time(x))


print(df.columns)

colunas = ['QUARTER', 'MONTH',
           'DAY_OF_WEEK','DAY_OF_MONTH','DEP_DEL15','ARR_DEL15',
           'ARR_TIME_BLOCK', 'DEP_TIME_BLOCK']
for col in colunas:
  df[col] = df[col].astype('category')

print(df.dtypes)

df['DISTANCE_cat'] = pd.qcut(df['DISTANCE'], q=8)



pd.DataFrame({'unicos':df.nunique(),
              'missing': df.isna().mean()*100,
              'tipo':df.dtypes})

# In[] some statstic

#delayed distrubution by time blok
arr_del15_distribution = df.groupby('ARR_TIME_BLOCK')['ARR_DEL15'].value_counts(normalize=True) * 100
arr_del15_counts = df.groupby('ARR_TIME_BLOCK')['ARR_DEL15'].value_counts()

time_blk = pd.concat([arr_del15_distribution, arr_del15_counts], axis=1)
time_blk.columns = ['Percentage', 'Count']

#delayed distrubution by day
week = data[['DAY_OF_WEEK','ARR_DEL15']].groupby('DAY_OF_WEEK').sum().sort_values(by='ARR_DEL15',ascending=False)
week['PERCENTUAL'] = week['ARR_DEL15']/(week['ARR_DEL15'].sum())*100
month = data[['DAY_OF_MONTH','ARR_DEL15']].groupby('DAY_OF_MONTH').sum().sort_values(by='ARR_DEL15',ascending=False)
month['PERCENTUAL'] = month['ARR_DEL15']/(month['ARR_DEL15'].sum())*100

print('>> Delayed flights by weekday<<')
print(week)
print('\n')
print('>> Delayed flights by monthday <<')
print(month)



# In[] catagorical to numeric

# dummies = pd.get_dummies(df[['ORIGIN', 'DEST']])
# dummies = dummies.astype(float)
# # Concatenate the dummies with the original Frame
# df = pd.concat([df, dummies], axis=1)

# # Drop the original 'ORIGIN' and 'DEST' columns
# df = df.drop(['ORIGIN', 'DEST'], axis=1)






# In[] Models preparation
df=df.drop(['DEP_TIME','ARR_TIME','DISTANCE'], axis=1)



cat_vars_final = df.select_dtypes(['object','category'])
cat_vars_final.drop(['ARR_DEL15','ARR_TIME_BLOCK','DEP_TIME_BLOCK'], axis=1, inplace=True)

enc = OneHotEncoder().fit(cat_vars_final)

cat_vars_ohe_final = enc.transform(cat_vars_final).toarray()
feature_names = enc.get_feature_names_out(cat_vars_final.columns.tolist())
cat_vars_ohe_final = pd.DataFrame(cat_vars_ohe_final, index=cat_vars_final.index, columns=feature_names)





y=df['ARR_DEL15']
X=cat_vars_ohe_final

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True)



# In[] decisionTree
tree1 = tree.DecisionTreeRegressor(criterion='squared_error')

tree1 = tree1.fit(X_train, y_train)
y_pred=tree1.predict(X_test)
result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
result['Match'] = result['Predicted'] == result['Actual']

print('Decision Tree R2 value')
print(r2_score(y_test, tree1.predict(X_test)))


# In[] random forest
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0) 

rf_reg.fit(X_train,y_train)
y_pred=rf_reg.predict(X_test)
y_pred = np.where(y_pred < 0.5, 0, 1)
result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
result['Match'] = result['Predicted'] == result['Actual']


cm = confusion_matrix(y_test, y_pred)
print('accurucy score:',accuracy_score(y_test, y_pred))


print(np.count_nonzero(y_test == 0))

print('Random Forest R2 value')
print(r2_score(y_test, rf_reg.predict(X_test)))

# In[] logistic regression
lr_model_final = LogisticRegression(C=1.0,n_jobs=-1,verbose=1, random_state=154)
lr_model_final.fit(X_train, y_train)

y_pred=lr_model_final.predict(X_test)
y_pred = np.where(y_pred < 0.5, 0, 1)
result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
result['Match'] = result['Predicted'] == result['Actual']
cm = confusion_matrix(y_test, y_pred)
print('accurucy score:',accuracy_score(y_test, y_pred))


cv = StratifiedKFold(n_splits=3, shuffle=True)
result = cross_val_score(lr_model_final,X,y, cv=cv, scoring='roc_auc')
print(f'A média: {np.mean(result)}')
print(f'Limite Inferior: {np.mean(result)-2*np.std(result)}')
print(f'Limite Superior: {np.mean(result)+2*np.std(result)}')




# In[] XGboost


sc=StandardScaler()


X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train_sc, y_train)

y_pred= classifier.predict(X_test_sc)

cm = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test, y_pred))








