import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.python.estimator import keras

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import tensorflow as tf
X = []
y = []
# TODO prepare your data here namely X (Pandas.Dataframe) and y ( Pandas.Series).( before feeding the model with data convert these to numpy array by using "to_numpy()" )
# you can try under and over sampling methods while preparing your data



# Split the data to test and traning data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42 , stratify= y) # stratify on y ( means that percentage of positive and negative instances are equal in training and testing set

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# a method returns the Deep Neural Network
# we are using a function because for k-cross-fold validation , we will create k models each of them will be trained with the different part of the training
# you can play with parameters of Dense layer ( check https://keras.io/api/layers/core_layers/dense/ )
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