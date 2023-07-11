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


# In[] data read
data=pd.read_csv('flightdata.csv')
df = pd.read_csv('flightdata.csv')
df20 = pd.read_csv('flightdata.csv')

# In[] visiulasition
import seaborn as sns

sns.distplot(df['ARR_DELAY'], bins=20, kde=True, hist_kws={"range": [-60, 60]})

# Set plot title and labels
plt.title("Distribution Plot of ARR_DELAY")
plt.xlabel("ARR_DELAY")
plt.ylabel("Frequency")
plt.xlim([-20,20])
# Show the plot
plt.show()


sns.catplot(y="ARR_DELAY", data=df, kind='box')
plt.ylim(-50, 40)
# Set plot title and labels
plt.title("Box Plot of ARR_DELAY")
plt.ylabel("ARR_DELAY")

# In[] data preprocessing
corr=df.corr()


#Converting DATETIME
df.rename(columns={'DAY_OF_MONTH': 'DAY'}, inplace=True)
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
df['TIME'] = pd.to_datetime(df['CRS_DEP_TIME'], format='%H%M').dt.time
df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))



#missing values
df.isna().sum()






# In[] adjusted time calc
# df['ADJUSTED']=df['ARR_DELAY']-df['DEP_DELAY']


# adjusted=df[df['ADJUSTED'] < 0]['ADJUSTED']



# In[] deleting dummy colums

print( df.columns)
dummycolumns=['QUARTER', 'MONTH', 'DAY', 'DAY_OF_WEEK',
              'DATE', 'TIME',
              'Unnamed: 25','YEAR','ARR_DEL15', 'ACTUAL_ELAPSED_TIME',
              'DEP_TIME', 'DEP_DELAY',
              'DEP_DEL15','ARR_TIME',
              'TAIL_NUM', 'FL_NUM','ORIGIN_AIRPORT_ID',
              'DEST_AIRPORT_ID','UNIQUE_CARRIER',
              'CRS_DEP_TIME','CRS_ARR_TIME']

for i in dummycolumns:
    try:
        df = df.drop(i, axis=1)
    except:
        print(i)
        continue
    
    
#df20 operations    

dummycolumns=[
              'DATE', 'TIME',
              'Unnamed: 25','YEAR', 'ACTUAL_ELAPSED_TIME',
              'DEP_TIME',
              'ARR_TIME',
              'TAIL_NUM', 'FL_NUM','ORIGIN_AIRPORT_ID',
              'DEST_AIRPORT_ID','UNIQUE_CARRIER',
              'CRS_DEP_TIME','CRS_ARR_TIME']


for i in dummycolumns:
    try:
        df20 = df20.drop(i, axis=1)
    except:
        print('this column can not be deleted: ',i)
        continue
        


# In[] catagorical to numeric

dummies = pd.get_dummies(df[['ORIGIN', 'DEST']])
dummies = dummies.astype(float)
# Concatenate the dummies with the original DataFrame
df = pd.concat([df, dummies], axis=1)

# Drop the original 'ORIGIN' and 'DEST' columns
df = df.drop(['ORIGIN', 'DEST'], axis=1)

# operation df20
dummies = pd.get_dummies(df20[['ORIGIN', 'DEST']])
dummies = dummies.astype(float)
# Concatenate the dummies with the original DataFrame
df20 = pd.concat([df20, dummies], axis=1)

# Drop the original 'ORIGIN' and 'DEST' columns
df20 = df20.drop(['ORIGIN', 'DEST'], axis=1)


# In[] Seperating data
#df2 canceled or diverted
#df3 delayed

filtered_df = df[(df['ARR_DELAY'].isnull()) & ((df['CANCELLED'] != 0) & (df['DIVERTED'] != 0))]

# Get the count of rows satisfying the criteria
row_count = filtered_df.shape[0]

df2=df[(df['ARR_DELAY'].isnull()) ]
df3=df[df['ARR_DELAY'].notnull()]


print(df3.isna().sum())

#df20 without datetime
filtered_df = df20[(df20['ARR_DELAY'].isnull()) & ((df20['CANCELLED'] != 0) & (df20['DIVERTED'] != 0))]

# Get the count of rows satisfying the criteria
row_count = filtered_df.shape[0]

df21=df20[(df20['ARR_DELAY'].isnull()) ]
df22=df20[df20['ARR_DELAY'].notnull()]

# In[]

#finding how many rows in certain range of time
count = len(df[(df['ARR_DELAY'] >= -15) & (df['ARR_DELAY'] <= 15)])


# In[] Models for delay

y=df22['ARR_DELAY']
X=df22.drop(['CANCELLED', 'DIVERTED','ARR_DELAY'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True)

# In[] decisionTree
tree1 = tree.DecisionTreeRegressor(criterion='squared_error')

tree1 = tree1.fit(X_train, y_train)
y_pred=tree1.predict(X_test)
result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test, 'diff':y_pred-y_test})

print('Decision Tree R2 value')
print(r2_score(y_test, tree1.predict(X_test)))


# In[] random forest
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0) 

rf_reg.fit(X_train,y_train)
y_pred=rf_reg.predict(X_test)
result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test, 'diff':y_pred-y_test})

y_pred_t = np.where((y_pred >= -15) & (y_pred <= 15), 0, 1)
y_test_t= np.where((y_test >= -15) & (y_test <= 15), 0, 1)

result_t = pd.DataFrame({'Predicted': y_pred_t, 'Actual': y_test_t, 'diff':y_pred_t-y_test_t})
cm = confusion_matrix(y_test_t, y_pred_t)
print(accuracy_score(y_test_t, y_pred_t))


print(np.count_nonzero(y_test_t == 0))

print('Random Forest R2 value')
print(r2_score(y_test, rf_reg.predict(X_test)))

# In[] lineear reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)
result = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test, 'diff':y_pred-y_test})

print('linear reg R2 value')
print(r2_score(y_test, lin_reg.predict(X_test)))

y_pred_t = np.where((y_pred >= -15) & (y_pred <= 15), 0, 1)
y_test_t= np.where((y_test >= -15) & (y_test <= 15), 0, 1)

result_t = pd.DataFrame({'Predicted': y_pred_t, 'Actual': y_test_t, 'diff':y_pred_t-y_test_t})
cm = confusion_matrix(y_test_t, y_pred_t)
print(accuracy_score(y_test_t, y_pred_t))

# In[] XGboost


sc=StandardScaler()


X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)
y_train_t= np.where((y_train >= -15) & (y_train <= 15), 0, 1)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train_sc, y_train_t)

y_pred_t = classifier.predict(X_test_sc)

cm = confusion_matrix(y_test_t, y_pred_t)

print(accuracy_score(y_test_t, y_pred_t))








