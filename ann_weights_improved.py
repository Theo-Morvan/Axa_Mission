# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:08:44 2019

@author: cmorv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
from imblearn.over_sampling  import SMOTE
import xgboost
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import time

raw_data = pd.read_csv('C:/Users/cmorv/AXA/ann_model/data_ann.csv')
raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
raw_data.head()


targets =  pd.get_dummies(raw_data['Nombre pieces principales']).values
inputs_stdsc =raw_data[['Valeur fonciere', 'Code type local',
                        'year', 'month', 'day', 'P15_POP', 'MED14','Part population dense (1)',
                        'Part population intermédiaire (2)', 'Part population peu dense (3)',
                        'Part population très peu dense (4)', 'Longitude', 'Latitude']].values
inputs_mmsc = raw_data[['nombre de redevables','Typo degré de densité']].values

sc = StandardScaler()
mms= MinMaxScaler()
inputs_stdsc = sc.fit_transform(inputs_stdsc)
inputs_mmsc = mms.fit_transform(inputs_mmsc)

scaled_inputs = np.concatenate((inputs_stdsc,inputs_mmsc),axis=1)

X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, targets, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(1/9))

y = y_train.argmax(axis=1)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y),
                                                  y)

# Convert class_weights to a dictionary to pass it to class_weight in model.fit
class_weights = dict(enumerate(class_weights))


dropout=0.3
classifier_unb = Sequential()
classifier_unb.add(Dense(units = 100, activation = 'relu', input_dim = X_train.shape[1]))
classifier_unb.add(Dropout(dropout))
classifier_unb.add(Dense(units = 100, activation = 'softmax'))
classifier_unb.add(Dropout(dropout))
classifier_unb.add(Dense(units = 100, activation = 'relu'))
classifier_unb.add(Dropout(dropout))
classifier_unb.add(Dense(units = 100, activation = 'softmax'))
classifier_unb.add(Dropout(dropout))
classifier_unb.add(Dense(units = 100, activation = 'relu'))
classifier_unb.add(Dropout(dropout))
classifier_unb.add(Dense(units = 6,activation = 'softmax'))


early_stop = EarlyStopping(monitor='val_loss', patience=2)
classifier_unb.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier_unb.fit(X_train, y_train, batch_size = 32, epochs = 7, 
               validation_data=(X_val,y_val), callbacks=[early_stop],
               verbose=2)
X_test_final = X_test
y_test_final = y_test


del X_train, X_test, X_val, y_val, y_train, y_test


smote = SMOTE()

X_data, y_data = smote.fit_resample(raw_data.iloc[:,1:],targets)

sc = StandardScaler()
mms= MinMaxScaler()
inputs_stdsc = sc.fit_transform(np.concatenate((X_data[:,:8], X_data[:,10:]), axis=1))
inputs_mmsc = mms.fit_transform(X_data[:,8:10])
scaled_inputs = np.concatenate((inputs_stdsc,inputs_mmsc),axis=1)

X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, y_data, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(1/9))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

number_tests = 10

y = y_train.argmax(axis=1)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y),
                                                  y)

# Convert class_weights to a dictionary to pass it to class_weight in model.fit
class_weights = dict(enumerate(class_weights))

x = np.linspace(-0.5,1,num=number_tests)

dropout=0.3
resultats = []
s=1
for valeur in range(number_tests):
    print("début de l'étape n°"+str(s))
    if s==1:
        print("début du programme à : ", 
              str(time.localtime()[3]), "h et ", str(time.localtime()[4]), "min" )
    class_dict = class_weights
    class_dict[2] += x[valeur]
    class_dict[3] +=(2.5)*x[valeur]
    class_dict[4] +=1/2*x[valeur]
    classifier = Sequential()
    classifier.add(Dense(units = 100, activation = 'relu', input_dim = X_train.shape[1]))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 100, activation = 'softmax'))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 100, activation = 'relu'))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 100, activation = 'softmax'))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 100, activation = 'relu'))
    classifier.add(Dropout(dropout))
    classifier.add(Dense(units = 6,activation = 'softmax'))
    early_stop = EarlyStopping(monitor='val_loss', patience=1)
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 5, 
                   validation_data=(X_val,y_val), callbacks=[early_stop],
                   class_weight=class_dict,
                   verbose=0)
    resultats.append(classifier.predict(X_test))
    print("fin de l'étape n°"+str(s))
    print("Fin de l'étape à ", str(time.localtime()[3]), "h et ", str(time.localtime()[4]), "min")
    s += 1


y_resultat_unb = classifier_unb.predict(X_test_final)
y_resultat_unb_weights = classifier_unb.predict(X_test_final)
y_resultat = (1/2)*(y_resultat_unb+y_resultat_unb_weights)
plot_confusion_matrix(y_test_final.argmax(axis=1), y_resultat_unb_weights.argmax(axis=1),
                          classes=[x for x in range(1,7)], normalize=True, title='Normalized confusion matrix')

for valeur in range(number_tests):
    plot_confusion_matrix(y_test.argmax(axis=1), resultats[valeur].argmax(axis=1),
                          classes=[x for x in range(1,7)], normalize=True, title='Normalized confusion matrix')

cf = classification_report(y_test.argmax(axis=1), y_resultat.argmax(axis=1))

def balance_weights(data, weights_to_imp, coefs):
    y = y_train.argmax(axis=1)
    class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y),
                                                  y)
    class_weights = dict(enumerate(class_weights))
    for couple in zip(weights_to_imp, coefs):
        class_weights[couple[0]] += couple[1]
    return class_weights

balance_weights(y_train, [4], [0.5])
