import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, categorical_column, label, scale_by_group = True):
        self.categorical_column = categorical_column
        self.label = label
        self.scale_by_group = scale_by_group
        self.scaler = MinMaxScaler()
        self.scaler_dict = {}
        
    def fit(self, X_train):
        if self.scale_by_group:
            groups_train = X_train.groupby(self.categorical_column)
            for name, group in groups_train:
                self.scaler = MinMaxScaler()
                self.scaler_dict[name] = self.scaler.fit(group[self.label].values.reshape(-1,1))       
        else:
            self.scaler.fit(X_train[self.label].values.reshape(-1,1))

    def transform(self, X):
        if self.scale_by_group:
            temp_dfs = []
            for name in self.scaler_dict.keys():
                temp = X[X[self.categorical_column] == name].copy()
                temp[self.label] = self.scaler_dict[name].transform(temp[self.label].values.reshape(-1,1)) 
                temp_dfs.append(temp)
            return pd.concat(temp_dfs)
        else:
            X[self.label] = self.scaler.transform(X[self.label].values.reshape(-1,1))
            return X

    def inverse_transform(self, X,label):
        if self.scale_by_group:
            temp_dfs = []
            for name in self.scaler_dict.keys():
                temp = X[X[self.categorical_column] == name].copy()
                temp[label] = self.scaler_dict[name].inverse_transform(temp[label].values.reshape(-1,1)) 
                temp_dfs.append(temp)
            return pd.concat(temp_dfs)
        else:
            X[label] = self.scaler.inverse_transform(X[label].values.reshape(-1,1))
            return X
        
    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
                             
class DataPreprocessor:
    def __init__(self,X_train):
        one_hot1 = np.eye(4)
        one_hot2 = np.eye(2)
        self.measurement_type_category = {category: one_hot2[i] for i, category in enumerate(np.unique(X_train['measurement_type']))}
        self.kinase_name_category = {category: one_hot1[i] for i, category in enumerate(np.unique(X_train['Kinase_name']))}       
        self.measurement_type_inv_map = {np.argmax(i):category for category, i in self.measurement_type_category.items()}
        self.kinase_name_inv_map = {np.argmax(i):category for category, i in self.kinase_name_category.items()}

    def transform(self, X):
        X['measurement_type'] = X['measurement_type'].map(self.measurement_type_category)
        X['Kinase_name'] = X['Kinase_name'].map(self.kinase_name_category)
        return X

    def reverse_transform(self, X):
        X['measurement_type'] = X['measurement_type'].apply(lambda x: np.argmax(x)) 
        X['Kinase_name'] = X['Kinase_name'].apply(lambda x: np.argmax(x)) 
        X['measurement_type'] = X['measurement_type'].map(self.measurement_type_inv_map)
        X['Kinase_name'] = X['Kinase_name'].map(self.kinase_name_inv_map)
        return X
    