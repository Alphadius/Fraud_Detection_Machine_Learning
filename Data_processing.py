# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:27:54 2021

@author: maubr
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_transaction = pd.read_csv('train_transaction.csv')
test_transaction = pd.read_csv('test_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')
test_identity = pd.read_csv('test_identity.csv')

#on drop les TransactionId car elle sont inutile pour la prediction
train_transaction = train_transaction.drop(columns='TransactionID')
test_transaction = test_transaction.drop(columns='TransactionID')
train_identity = train_identity.drop(columns='TransactionID')
test_identity = test_identity.drop(columns='TransactionID')

#ce que l'on veut predire notre target
y_train = train_transaction["isFraud"]

#on l'enleve de notre model d'entrainement sinon on aurais un model overfit 
train_transaction = train_transaction.drop(columns = ["isFraud"])

#on met tout les nom des collones au même format 
test_identity.columns = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
       'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
       'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
       'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
       'DeviceInfo']

#on cancat les fichier test et train pour permetre de faire les operation sur 2 fichiers au lieux de 4 
transaction_data = pd.concat([train_transaction, test_transaction])
identity_data = pd.concat([train_identity, test_identity])


del train_identity
del test_identity
del test_transaction
del train_transaction

#on met dans un liste les colonnes categorique dans cat_id_cols et numerical dans num_id_cols  
c = (identity_data.dtypes == 'object')
n = (identity_data.dtypes != 'object')
cat_id_cols = list(c[c].index)
num_id_cols = list(n[n].index) 

#pareil pour transaction  
c = (transaction_data.dtypes == 'object')
n = (transaction_data.dtypes != 'object')
cat_trans_cols = list(c[c].index)
num_trans_cols = list(n[n].index) 


#on regarde combien de valeur manquante categorique il y a dans le fichier 
#identity
low_missing_cat_id_cols = []      # en dessous de 15% de valeur manquante 
medium_missing_cat_id_cols = []   # entre 15% et 60% de valeur manquante 
many_missing_cat_id_cols = []     # plus de 60% de valeur manquante 

for i in cat_id_cols:
    percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])
    if percentage < 15:
        low_missing_cat_id_cols.append(i)
    elif percentage >= 15 and percentage < 75:
        medium_missing_cat_id_cols.append(i)
    else:
        many_missing_cat_id_cols.append(i)
        
#pour les valeurs numeriques
low_missing_num_id_cols = []      # en dessous de 15% de valeur manquante 
medium_missing_num_id_cols = []   # entre 15% et 60% de valeur manquante 
many_missing_num_id_cols = []     # plus de 60% de valeur manquante 

for i in num_id_cols:
    percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])
    if percentage < 15:
        low_missing_num_id_cols.append(i)
    elif percentage >= 15 and percentage < 75:
        medium_missing_num_id_cols.append(i)
    else:
        many_missing_num_id_cols.append(i)
        

#la même chose pour transaction
low_missing_num_trans_cols = []      #  en dessous de 15% de valeur manquante 
medium_missing_num_trans_cols = []   # entre 15% et 60% de valeur manquante 
many_missing_num_trans_cols = []     # more than 60% missing

for i in num_trans_cols:
    percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])
    if percentage < 15:
        low_missing_num_trans_cols.append(i)
    elif percentage >= 15 and percentage < 75:
        medium_missing_num_trans_cols.append(i)
    else:
        many_missing_num_trans_cols.append(i)
        
print("num_trans_cols: \n\n")        
print("number low missing: ", len(low_missing_num_trans_cols), "\n")
print("number medium missing: ", len(medium_missing_num_trans_cols), "\n")
print("number many missing: ", len(many_missing_num_trans_cols), "\n")

low_missing_cat_trans_cols = []      #  en dessous de 15% de valeur manquante 
medium_missing_cat_trans_cols = []   # entre 15% et 60% de valeur manquante 
many_missing_cat_trans_cols = []     # plus de 60% de valeur manquante 

for i in cat_trans_cols:
    percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])
    if percentage < 15:
        low_missing_cat_trans_cols.append(i)
    elif percentage >= 15 and percentage < 75:
        medium_missing_cat_trans_cols.append(i)
    else:
        many_missing_cat_trans_cols.append(i)
        
#on supprime toute les colonnes qui on plus de 60% de valeur manquante pour les valeurs numeriques
transaction_data = transaction_data.drop(columns = many_missing_num_trans_cols)
      
identity_data = identity_data.drop(columns = many_missing_num_id_cols)



#comme on vient de supprimer des colonnes on refais le liste des colonnes numeriques du fichier
n = (transaction_data.dtypes != 'object')
num_trans_cols = list(n[n].index) 

#pareil pour identity
n = (identity_data.dtypes != 'object')
num_id_cols = list(n[n].index) 

#on fait la moyenne sur les colonnes qui ont moins de 15% de valeurs manquantes 
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy = 'mean') 
my_imputer.fit(transaction_data[low_missing_num_trans_cols])
transaction_data[low_missing_num_trans_cols] = my_imputer.transform(transaction_data[low_missing_num_trans_cols])


#la même chose pour identity
my_imputer = SimpleImputer(strategy = 'mean') 
my_imputer.fit(identity_data[low_missing_num_id_cols])
identity_data[low_missing_num_id_cols] = my_imputer.transform(identity_data[low_missing_num_id_cols])


#on fait la median sur les colonnes qui ont entre 15% et 60% de valeurs manquantes 
my_imputer = SimpleImputer(strategy = 'median') 
my_imputer.fit(transaction_data[medium_missing_num_trans_cols])
transaction_data[medium_missing_num_trans_cols] = my_imputer.transform(transaction_data[medium_missing_num_trans_cols])


#la même chose pour identity
my_imputer = SimpleImputer(strategy = 'median') 
my_imputer.fit(identity_data[medium_missing_num_id_cols])
identity_data[medium_missing_num_id_cols] = my_imputer.transform(identity_data[medium_missing_num_id_cols])

#CAT DATA
#on supprime toute les colonnes qui on plus de 60% de valeur manquante pour les valeurs catgoriques
transaction_data = transaction_data.drop(columns = many_missing_cat_trans_cols)
identity_data = identity_data.drop(columns = many_missing_cat_id_cols)

#comme on vient de supprimer des colonnes on refais le liste des colonnes categoriques du fichier
c = (transaction_data.dtypes == 'object')
cat_trans_cols = list(c[c].index) 

c = (identity_data.dtypes == 'object')
cat_id_cols = list(c[c].index)


low_card_trans_cols = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M6", "M7", "M8", "M9"]
high_card_trans_cols = ["P_emaildomain"]



for i in cat_trans_cols:
    most_frequent_value = transaction_data[i].mode()[0]
    print("For column: ", i, "the most frequent value is: ", most_frequent_value, "\n")
    transaction_data[i].fillna(most_frequent_value, inplace = True)

from sklearn.preprocessing import LabelEncoder
    
label_encoder = LabelEncoder()
print("transaction_data.shape before label-encoding: ", transaction_data.shape, "\n")

transaction_data[high_card_trans_cols] = label_encoder.fit_transform(transaction_data[high_card_trans_cols])

print("transaction_data.shape after label-encoding: ", transaction_data.shape, "\n")
print("transaction_data[high_card_trans_cols] after label_encoding: ",transaction_data[high_card_trans_cols], "\n")

#Identity
for col in cat_id_cols:
    print(col, identity_data[col].nunique(), "\n")

low_card_id_cols =  ["id_12", "id_15", "id_16", "id_28", "id_29", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType"]
high_card_id_cols = ["id_30", "id_31", "id_33", "DeviceInfo"]
    
print("lists successfully created!")

for i in cat_id_cols:
    most_frequent_value = identity_data[i].mode()[0]
    print("For column: ", i, "the most frequent value is: ", most_frequent_value, "\n")
    identity_data[i].fillna(most_frequent_value, inplace = True)

label_encoder = LabelEncoder()

print("identity_data.shape before label-encoding: ", identity_data.shape, "\n")

for col in high_card_id_cols:
    identity_data[col] = label_encoder.fit_transform(identity_data[col])

print("identity_data.shape after label-encoding: ", identity_data.shape, "\n")
print("identity_data[high_card_id_cols] after label_encoding: ",identity_data[high_card_id_cols], "\n")

print(transaction_data.info())
print(identity_data.info())


#LOW CARDINALITE CAT DATA
print("shape before encoding: ", transaction_data.shape, "\n")
print("columns to encode: ", low_card_trans_cols, "\n")
print("transaction_data.columns.to_list() before encoding: ", transaction_data.columns.to_list(), "\n")


# this line does the onehot encoding
low_card_trans_encoded = pd.get_dummies(transaction_data[low_card_trans_cols], dummy_na = False)
transaction_data.drop(columns = low_card_trans_cols, inplace = True)

print("shape after encoding: ", transaction_data.shape, "\n\n")
print("shape of new dataframe: ", low_card_trans_encoded.shape, "\n\n")
print("newly generated columns: ", low_card_trans_encoded.columns, "\n")
print("low_card_trans_encoded.info(): ", low_card_trans_encoded.info(),"\n")
print("transaction_data.columns.to_list() after encoding: ", transaction_data.columns.to_list(), "\n")


print("shape before encoding: ", identity_data.shape, "\n")
print("columns to encode: ", low_card_id_cols, "\n")

# this line does the onehot encoding
low_card_id_encoded = pd.get_dummies(identity_data[low_card_id_cols], dummy_na = False)
identity_data.drop(columns = low_card_id_cols, inplace = True)


print("shape after encoding: ", identity_data.shape, "\n\n")
print("shape of new dataframe: ", low_card_id_encoded.shape, "\n\n")
print("newly generated columns: ", low_card_id_encoded.columns, "\n")
print("low_card_id_encoded.info(): ", low_card_id_encoded.info())

print(transaction_data.isnull().sum().sum(), "\n")
print(low_card_trans_encoded.isnull().sum().sum())

print(transaction_data.info(), "\n")
print(low_card_trans_encoded.info())

print(identity_data.info(), "\n")
print(low_card_id_encoded.info())

print("transaction_data.shape before concatting: ", transaction_data.shape, "\n")
print("low_card_trans_encoded.shape before concatting: ", low_card_trans_encoded.shape, "\n")

transaction_concatted = pd.concat([transaction_data, low_card_trans_encoded], axis = 1)

print("transaction_concatted.shape after concatting: ", transaction_concatted.shape, "\n")
print("transaction_concatted.columns after concatting: ", transaction_concatted.columns, "\n")

#del low_card_trans_encoded
#del transaction_data

print(transaction_concatted.info())

print("identity_data.shape before concatting: ", identity_data.shape, "\n")
print("low_card_id_encoded.shape before concatting: ", low_card_id_encoded.shape, "\n")

identity_concatted = pd.concat([identity_data, low_card_id_encoded], axis = 1)

print("identity_concatted.shape after concatting: ", identity_concatted.shape, "\n")
print("identity_concatted.columns after concatting: ", identity_concatted.columns, "\n")

#del low_card_id_encoded
#del identity_data

print(identity_concatted.info())
#creation du train_data et test_data

print("transaction_concatted.shape before splitting up: ", transaction_concatted.shape, "\n")

# shape of train_transaction was (590540, 393), 
# shape of test_transaction  was (506691, 392)
train_transaction = transaction_concatted.iloc[0:590540]
test_transaction = transaction_concatted.iloc[590540:]

print("train_transaction.shape after splitting up: ", train_transaction.shape, "\n")
print("test_transaction.shape after splitting up: ", test_transaction.shape, "\n")

print("identity_concatted.shape before splitting up: ", identity_concatted.shape, "\n")

# shape of train_identity was  (144233, 40)
# shape of test_identity  was  (141907, 40)
train_identity = identity_concatted.iloc[0:144233]
test_identity = identity_concatted.iloc[144233:]

print("train_identity.shape after splitting up: ", train_identity.shape, "\n")
print("test_identity.shape after splitting up: ", test_identity.shape, "\n")

print("train_transaction.shape before concatting: ", train_transaction.shape, "\n")
print("train_identity.shape before concatting: ", train_identity.shape, "\n")

train_data  = pd.concat([train_transaction, train_identity], axis = 1)

print("train_data.shape: ", train_data.shape)

counter = 0

for i in train_data.columns:
    
    summ = train_data[i].isnull().sum()
    print(i, summ)
    if summ > 0:
        counter += 1
        
print("\n number of columns with missing values: ", counter)

print("test_transaction.shape before concatting: ", test_transaction.shape, "\n")
print("test_identity.shape before concatting: ", test_identity.shape, "\n")

test_data  = pd.concat([test_transaction, test_identity], axis = 1)

print("test_data.shape: ", test_data.shape)

counter = 0

for i in train_data_Hyper.columns:
    
    summ = train_data_Hyper[i].isnull().sum()
    print(i, summ)
    if summ > 0:
        counter += 1
        
print("\n number of columns with missing values: ", counter)

test_data.to_csv(r'C:\Users\maubr\Desktop\Moi\COURS\M1\APPAUTO\PROJET\Python\EXPL\test_data75.csv', index = False)
train_data.to_csv(r'C:\Users\maubr\Desktop\Moi\COURS\M1\APPAUTO\PROJET\Python\EXPL\train_data75.csv', index = False)
y_train.to_csv(r'C:\Users\maubr\Desktop\Moi\COURS\M1\APPAUTO\PROJET\Python\EXPL\y_train75.csv', index = False)
train_identity_ID.to_csv(r'C:\Users\maubr\Desktop\LES DATA2\train_identity_ID.csv', index = False)
test_identity_ID.to_csv(r'C:\Users\maubr\Desktop\LES DATA2\test_identity_ID.csv', index = False)
train_transaction_ID.to_csv(r'C:\Users\maubr\Desktop\LES DATA2\train_transaction_ID.csv', index = False)
test_transaction_ID.to_csv(r'C:\Users\maubr\Desktop\LES DATA2\test_transaction_ID.csv', index = False)

test_data.fillna(test_data.median(), inplace=True)
train_data.fillna(train_data.median(), inplace=True)









