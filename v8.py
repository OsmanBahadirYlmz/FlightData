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

from tensorflow.python.estimator import keras
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import category_encoders as ce

def printScore(y_test,y_pred):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Display the results
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1-score: {:.3f}".format(f1))
    print("Accuracy: {:.3f}".format(accuracy))
    return

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

dummycolumns=['YEAR',
       'UNIQUE_CARRIER', 'FL_NUM', 'ORIGIN_AIRPORT_ID',
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

#tail number catagory encoder
tenc=ce.TargetEncoder() 
df['TAIL_DELAY']=tenc.fit_transform(df['TAIL_NUM'],df['ARR_DEL15'])

df['TAIL_DELAY2'] = df.groupby('TAIL_NUM')['ARR_DEL15'].transform('mean')

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

df['CRS_DEP_TIME'] = df['CRS_DEP_TIME'].astype('int')
df['CRS_DEP_TIME_BLOCK'] = df['CRS_DEP_TIME'].apply(lambda x :arr_time(x))

df['CRS_ARR_TIME'] = df['ARR_TIME'].astype('int')
df['CRS_ARR_TIME_BLOCK'] = df['CRS_ARR_TIME'].apply(lambda x :arr_time(x))

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

month2= data[['MONTH','ARR_DEL15']].groupby('MONTH').sum().sort_values(by='ARR_DEL15',ascending=False)
month2['PERCENTUAL'] = month2['ARR_DEL15']/(month['ARR_DEL15'].sum())*100


print('>> Delayed flights by weekday<<')
print(week)
print('\n')
print('>> Delayed flights by monthday <<')
print(month)


#graph week
weekday_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

week.index = pd.Series(week.index).map(weekday_map)

week.plot(kind='bar', y='PERCENTUAL', legend=None)
plt.xlabel('Day of the Week')
plt.ylabel('Arrival Delay Percentage')
plt.title('Arrival Delay Percentage by Day of the Week')
plt.ylim(0, max(week['PERCENTUAL']) * 1.1)

# Add bar values
for i, v in enumerate(week['PERCENTUAL']):
    plt.annotate(f"{v:.1f}", xy=(i, v), ha='center', va='bottom', color='green', fontsize='large')

# Display the graph
plt.show()

#graph day of month
plt.figure(figsize=(12, 20))
month.plot(kind='bar', y='PERCENTUAL', legend=None,width=0.8)
plt.xlabel('Day Of Month')
plt.ylabel('Arrival Delay Percentage')
plt.title('Arrival Delay Percentage by Day Of Month')
plt.ylim(0, max(month['PERCENTUAL']) * 1.1)
plt.tight_layout() 
# Add bar values
for i, v in enumerate(month['PERCENTUAL']):
    plt.annotate(f"{v:.1f}", xy=(i, v), ha='center', va='bottom', color='green', fontsize='large')
 # Automatically adjusts the spacing

# graph month
plt.figure(figsize=(12, 20))
month2.plot(kind='bar', y='PERCENTUAL', legend=None,width=0.8)
plt.xlabel('Month')
plt.ylabel('Arrival Delay Percentage')
plt.title('Arrival Delay Percentage Month')
plt.ylim(0, max(month2['PERCENTUAL']) * 1.1)
plt.tight_layout() 
for i, v in enumerate(month2['PERCENTUAL']):
    plt.annotate(f"{v:.1f}", xy=(i, v), ha='center', va='bottom', color='red', fontsize='large')
    
    

#graph for time block
dep_time_blok= df[['DEP_TIME_BLOCK','ARR_DEL15']].groupby('DEP_TIME_BLOCK').sum()
dep_time_blok= df[['DEP_TIME_BLOCK','ARR_DEL15']].groupby('DEP_TIME_BLOCK').sum().sort_values(by='ARR_DEL15',ascending=False)
dep_time_blok['PERCENTUAL'] = dep_time_blok['ARR_DEL15']/(dep_time_blok['ARR_DEL15'].sum())*100

print(df.dtypes)

dep_time_block_df = df.groupby('DEP_TIME_BLOCK')['ARR_DEL15'].agg(['count'])
print(dep_time_block_df['count'].sum())
# Renaming the columns
dep_time_block_df.columns = ['Count', 'Percentage']
dep_time_block_df = dep_time_block_df.sort_values('Percentage', ascending=False)

# Sorting the DataFrame by the index
dep_time_block_df = dep_time_block_df.sort_index()

# In[] catagorical to numeric

# dummies = pd.get_dummies(df[['ORIGIN', 'DEST']])
# dummies = dummies.astype(float)
# # Concatenate the dummies with the original Frame
# df = pd.concat([df, dummies], axis=1)

# # Drop the original 'ORIGIN' and 'DEST' columns
# df = df.drop(['ORIGIN', 'DEST'], axis=1)






# In[] Models preparation
# df=df.drop(['DEP_TIME','ARR_TIME','DISTANCE','CRS_DEP_TIME','DEP_DELAY','DEP_DEL15'], axis=1)

# df=df.drop(['DEP_TIME_BLOCK', 'ARR_TIME_BLOCK'], axis=1)

# df=df.drop(['CRS_ARR_TIME'], axis=1)

cat_vars_final = df.select_dtypes(['object','category'])
cat_vars_final.drop(['ARR_DEL15','TAIL_NUM'], axis=1, inplace=True)
cat_vars_final.drop(['DEP_TIME','ARR_TIME','DISTANCE','CRS_DEP_TIME','DEP_DELAY','DEP_DEL15'], axis=1, inplace=True)
cat_vars_final.drop(['DEP_TIME_BLOCK', 'ARR_TIME_BLOCK'], axis=1, inplace=True)
cat_vars_final.drop(['CRS_ARR_TIME'], axis=1, inplace=True)
cat_vars_final.drop(['DEP_DEL15'], axis=1, inplace=True)

enc = OneHotEncoder().fit(cat_vars_final)
cat_vars_ohe_final = enc.transform(cat_vars_final).toarray()
feature_names = enc.get_feature_names_out(cat_vars_final.columns.tolist())
cat_vars_ohe_final = pd.DataFrame(cat_vars_ohe_final, index=cat_vars_final.index, columns=feature_names)

cat_vars_ohe_final['TAIL_DELAY2']=df['TAIL_DELAY2']



y=df['ARR_DEL15']
X=cat_vars_ohe_final

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify= y, shuffle=True)

print(cat_vars_final.columns)

# In[] decisionTree
tree1 = tree.DecisionTreeRegressor(criterion='squared_error')

tree1 = tree1.fit(X_train, y_train)
y_pred=tree1.predict(X_test)
y_pred = np.where(y_pred < 0.5, 0, 1)

cm = confusion_matrix(y_test, y_pred)
printScore(y_test,y_pred)

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
printScore(y_test,y_pred)





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


# In[] Keras

def create_model():
    # creates Sequential model
    model = tf.keras.Sequential()
    # len(X_train.columns) is the number of columns in the X
    model.add(tf.keras.layers.Dense(len(X_train.columns)))
    # dense layer with 32 nodes uses activation function "relu" ( you can try different activation functions, but usually relu works the best ) you can also try to put "kernel_regularization" parameter
    model.add(tf.keras.layers.Dense(32 , activation='relu'))
    # dense layer with 64 nodes uses activation function "relu"
    model.add(tf.keras.layers.Dense(64 , activation='relu'))
    # dense layer with 128 nodes uses activation function "relu"
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # dense layer with 1 node -> 0 or 1
    model.add(tf.keras.layers.Dense(1 , activation='sigmoid'))
    # compiles the model by using "binary_crossentropy" which is the metric used for binary classification. optimizer is Adam as default. And the metric is 'accuracy'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # returns the model
    return model

# K-fold Cross Validation model evaluation


# prepare the folders
k_folder = 10
skf = StratifiedKFold(n_splits=k_folder, shuffle=True )

acc_per_fold = []
loss_per_fold = []
models = []

# you can play with these numbers

epoch = 50 # number of epochs
batch_size = 8 # batch size

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True ) # if there is no improvement in terms of val_loss the trainin stops and restores the best weights



# in "skf" traning set split into 10 files , this for loop iterates over folders and each iteration it uses 9 folders as training data and 1 folder as validation data
fold_no = 1

for k, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

    # gets the rows for training set and convert it to numpy array
    X_new_train = X.filter(items=train_index, axis=0).to_numpy()
    y_new_train = y.filter(items=train_index, axis=0).to_numpy()

    # gets the rows for validation set and convert it to numpy array
    X_new_val = X.filter(items=val_index, axis=0).to_numpy()
    y_new_val = y.filter(items=val_index, axis=0).to_numpy()

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # creates model
    model = create_model()
    # fits the data to model
    history = model.fit(X_new_train, y_new_train, epochs=epoch, batch_size =batch_size, validation_data=(X_new_val, y_new_val),callbacks=[callback], use_multiprocessing = True, verbose=2)


    # evaluates model on "testing set"
    scores = model.evaluate(X_test, y_test, verbose=0)

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    models.append(model) # save model
    acc_per_fold.append(scores[1] * 100) # save accuracy
    loss_per_fold.append(scores[0]) # save loss

    fold_no = fold_no + 1


max_acc_index = acc_per_fold.index(max(acc_per_fold)) # checks the index of max accuracy in acc_per_fold
final_model = models[max_acc_index] # selects the model with max accuracy on "testing set"


print(np.array(acc_per_fold).mean()) # average accuracy of all 10 created models

# Now you can save the final model and use it to get confusion matrix. If you will compare the confision matrix . ( but i suggest you to use avarage accuracy.

y_pred=final_model.predict(X_test)
y_pred = np.where(y_pred < 0.62, 0, 1)



cm = confusion_matrix(y_test, y_pred)
printScore(y_test,y_pred)

a=[]
for i in range(0,100,1):
    i/=100
    y_pred2 = np.where(y_pred < i, 0, 1)
    accuracy = accuracy_score(y_test, y_pred2)
    a.append(accuracy)

#save and load
final_model.save('model3')
# new_model = tf.keras.models.load_model('model')

# In[] lineear reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred=lin_reg.predict(X_test)

y_pred = np.where(y_pred < 0.5, 0, 1)
cm = confusion_matrix(y_test, y_pred)
printScore(y_test,y_pred)




