"""
Created on Thu Feb 18 21:39:54 2021

@author: romainloirs
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

test_data = pd.read_csv('test_data.csv')
train_data = pd.read_csv('train_data.csv')
y_train1 = pd.read_csv('y_train.csv')['isFraud']



            
correlation_matrix_test = test_data.corr()
correlated_features_test = set()            
for i in range(len(correlation_matrix_test.columns)):
    for j in range(i):
        if abs(correlation_matrix_test.iloc[i, j]) > 0.7:
            colname = correlation_matrix_test.columns[i]
            correlated_features_test.add(colname)            


train_data.drop(labels=correlated_features_test, axis=1, inplace=True)

test_data.drop(labels=correlated_features_test, axis=1, inplace=True)

#on remplace tou les NaN par la median
test_data.fillna(test_data.median(), inplace=True)
train_data.fillna(train_data.median(), inplace=True)


#undersampling and oversampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

undersample = RandomUnderSampler()
oversample = SMOTE()

steps = [("u",undersample),("o", oversample)]
pipeline = Pipeline(steps = steps)
train_data, y_train1 = pipeline.fit_resample(train_data,y_train1)

#RECHERCHE DES HYPER PARAMETRE 1 AVEC GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 23, 25, 28],
    'max_features': [2],
    'min_samples_leaf': [4],
    'min_samples_split': [2],
    'n_estimators': [100, 200, 500, 1000, 1500]
}

gridF = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2, scoring='accuracy')
bestF = gridF.fit(train_data, y_train1)

# afficher les paramètres optimaux
print("Les parametre optimaux pour le Random forest sont {} avec un score de {:.2f}".format(bestF.best_params_, bestF.best_score_))


#RECHERCHE DES HYPER PARAMETRE 2 AVEC HPSKLEARN
#on est obliger de transformer les Dataframe en nympy arrays pour que cette methode marche
test_datanp = test_data.to_numpy()
train_datanp = train_data.to_numpy()
y_train1np = y_train1.to_numpy()

#On utilise HyperoptEstimator de hpsklearn
from hpsklearn import random_forest
from hpsklearn import HyperoptEstimator
from hyperopt import tpe
model = HyperoptEstimator( classifier = random_forest('rf.random_forest'),
                            algo=tpe.suggest, 
                            max_evals=5, 
                            trial_timeout=500)

model.fit(train_datanp,y_train1np)

# afficher les paramètres optimaux
print(model.best_model())

#comme ont a trouve aucune amelioration avec les hyperparametres ont a pris la decsion de ne pas en mettre (c'est ce qui nous a donnez le meilleur score)

rf = RandomForestClassifier()                  
modelOpt = rf.fit(train_data, y_train1)
y_pred = modelOpt.predict_proba(test_data)

test_transaction = pd.read_csv('test_transaction.csv')
y_pred = pd.DataFrame(y_pred)
df1 = pd.DataFrame(y_pred,columns = ['isFraud'])
df1['isFraud'] = y_pred[1]
test1 = test_transaction["TransactionID"]
df2 = pd.DataFrame(data=test1,columns=['TransactionID'])
test =  pd.concat([df2, df1], axis=1)
test.to_csv(r'C:\Users\maubr\Desktop\Moi\COURS\M1\testRT12.csv', index = False)




fpr_cv_gauss, tpr_cv_gauss, thr_cv_gauss = metrics.roc_curve(y_test, y_pred)

# calculer l'aire sous la courbe ROC du modèle optimisé
auc_cv_gauss = metrics.auc(fpr_cv_gauss, tpr_cv_gauss)

# créer une figure
fig = plt.figure(figsize=(20, 20))

# afficher la courbe ROC du modèle optimisé
plt.plot(fpr_cv_gauss, tpr_cv_gauss, '-', lw=2, label='AUC=%.2f' % \
         ( auc_cv_gauss))
         

# donner un titre aux axes et au graphique
plt.xlabel('False Positive Rate', fontsize=30)
plt.ylabel('True Positive Rate', fontsize=30)
plt.title('SVM ROC Curve for a gaussian kernel', fontsize=30)

# afficher la légende
plt.legend(loc="lower right", fontsize=25)

# afficher l'image
plt.show()

fig = plt.figure(figsize=(20, 20))
plot_confusion_matrix(forestOpt, X_test, y_test)  
plt.title('confusion matrix for a gaussian kernel', fontsize=25)
plt.show()  
