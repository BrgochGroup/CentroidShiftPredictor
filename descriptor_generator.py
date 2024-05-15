# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:00:41 2020

@author: Ya Zhuo, University of Houston

Last modified on May 15 12:34:45 2024
@author: Amit Kumar, University of Houston
"""

## create an excel file named c_pounds.xlsx, in which the compositions that you want to predict are listed in the first column with the header "Formula"

# To run this code you need three Excel files;

## 1.c_pounds.xlsx   2. elements.xlsx  3. relative_permittivity_training_set.xlsx


import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from io import BytesIO
import os

class Vectorize_Formula:
    def __init__(self, elem_df):
        self.element_df = elem_df
        self.element_df.set_index('Symbol', inplace=True)
        self.column_names = []
        for string in ['avg', 'diff', 'max', 'min', 'std']:
            for column_name in list(self.element_df.columns.values):
                self.column_names.append(string + '_' + column_name)

    def get_features(self, formula):
        try:
            comp = Composition(formula)
            fractional_composition = comp.fractional_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            for key, value in fractional_composition.items():
                avg_feature += self.element_df.loc[key].values * value

            diff_feature = self.element_df.loc[list(fractional_composition.keys())].max() - self.element_df.loc[list(fractional_composition.keys())].min()
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            std_feature = self.element_df.loc[list(fractional_composition.keys())].std(ddof=0)

            features = np.concatenate([avg_feature, diff_feature, max_feature, min_feature, std_feature])
            return features
        except Exception as e:
            print(f'There was an error with the Formula: {formula}. Error: {e}')
            return [np.nan] * len(self.element_df.iloc[0]) * 5


element_df = pd.read_excel('elements.xlsx') # element file

formula_df = pd.read_excel('c_pounds.xlsx', sheet_name='Sheet1')  # Compostions file 

gf = Vectorize_Formula(element_df)

features = []
for formula in formula_df['Formula']:  
    features.append(gf.get_features(formula))

X = pd.DataFrame(features, columns=gf.column_names)

composition = formula_df[['Formula']]  
predicted = pd.concat([composition, X], axis=1)
header = ["Composition"] + gf.column_names
predicted.to_excel('to_predict.xlsx', index=False, header=header)


df = pd.read_excel('to_predict.xlsx')
df.columns = [col.replace('gilmor number of valence electron', 'gilman number of valence electron') for col in df.columns]
df.columns = [col.replace('heat atomization', 'enthalpy of atomization (kJ/mol)') for col in df.columns]
df.to_excel('modified_to_predict_1.xlsx', index=False)

df_large = pd.read_excel('modified_to_predict_1.xlsx')
df_small = pd.read_excel('relative_permittivity_training_set.xlsx') #training data file
common_columns = df_large.columns.intersection(df_small.columns)
df_filtered = df_large[common_columns]
c_pounds = pd.read_excel('c_pounds.xlsx', usecols=[0]) 
df_filtered.insert(0, c_pounds.columns[0], c_pounds.iloc[:, 0])

df_filtered.to_excel('to_predict_relative_permittivity.xlsx', index=False)

files_to_delete = ['to_predict.xlsx', 'modified_to_predict_1.xlsx']

for file_path in files_to_delete:
    if os.path.exists(file_path):
        os.remove(file_path)
        
    else:
        print(f"File not found: {file_path}")

print(f"Columns in filtered DataFrame: {len(df_filtered.columns)}")
print("A file named to_predict_relative_permittivity.xlsx has been generated.\nPlease check your folder.")

## Note-13 additional features are structural features that you have to find manually
