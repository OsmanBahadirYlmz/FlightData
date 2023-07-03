import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

data=pd.read_csv('flightdata.csv')
df = pd.read_csv('flightdata.csv')

#visiulasition
import seaborn as sns

sns.distplot(df['ARR_DELAY'], bins=20, kde=True, hist_kws={"range": [-60, 60]})

# Set plot title and labels
plt.title("Distribution Plot of ARR_DELAY")
plt.xlabel("ARR_DELAY")
plt.ylabel("Frequency")
plt.xlim([-60,60])
# Show the plot
plt.show()


sns.catplot(y="ARR_DELAY", data=df, kind='box')
plt.ylim(-50, 40)
# Set plot title and labels
plt.title("Box Plot of ARR_DELAY")
plt.ylabel("ARR_DELAY")



df.isna().sum()


filtered_df = df[(df['ARR_DEL15'].isnull()) & ((df['CANCELLED'] != 0) & (df['DIVERTED'] != 0))]

# Get the count of rows satisfying the criteria
row_count = filtered_df.shape[0]




df['ADJUSTED']=df['ARR_DELAY']-df['DEP_DELAY']

corr=df.corr()
adjusted=df[df['ADJUSTED'] < 0]['ADJUSTED']




print( df.columns)
dummycolumns=['Unnamed: 25','YEAR','ARR_DEL15', 'ACTUAL_ELAPSED_TIME',
              'DEP_TIME', 'DEP_DELAY',
              'DEP_DEL15','ARR_TIME',
              'TAIL_NUM', 'FL_NUM','ORIGIN_AIRPORT_ID',
              'DEST_AIRPORT_ID','UNIQUE_CARRIER']

for i in dummycolumns:
    try:
        df = df.drop(i, axis=1)
    except:
        print(i)
        continue

df2=df[(df['ARR_DELAY'].isnull()) ]
df3=df[df['ARR_DELAY'].notnull()]


print(df3.isna().sum())

# catagorical to numeric

dummies = pd.get_dummies(df3[['ORIGIN', 'DEST']])

# Concatenate the dummies with the original DataFrame
merged = pd.concat([df3, dummies], axis=1)

# Drop the original 'ORIGIN' and 'DEST' columns
merged = merged.drop(['ORIGIN', 'DEST'], axis=1)

y=merged['ARR_DELAY']
X=merged.drop(['CANCELLED', 'DIVERTED','ADJUSTED','ARR_DELAY'], axis=1)








