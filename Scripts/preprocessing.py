import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


import warnings
warnings.filterwarnings('ignore')


class CustomColTransfomer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X, y=None):
        X["Origin"] = X["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
        return X

def num_pipeline_transformer():
    '''
    Function to process numerical transformations
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''

    num_attrs = ['Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    return num_attrs, num_pipeline

def cat_pipeline_transformer():
    '''
    Function to process categorical transformations
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''

    cat_attrs = ["Origin"]

    cat_pipeline = Pipeline ([
        ('col_map', CustomColTransfomer()),
        ('encoder', OneHotEncoder())
    ])

    return cat_attrs, cat_pipeline


def preprocess(df):
    '''
    Function to process an entire dataset
    Returns:
        X_train: preprocessed trainning attributes
        y_train: label column
        
    '''

    X_train = df.drop("MPG", axis=1)
    y_train = df["MPG"].copy()

    num_attrs, num_pipeline = num_pipeline_transformer()
    cat_attrs, cat_pipeline = cat_pipeline_transformer()

    final_attrs = ['Germany', 'India', 'USA'] + num_attrs

    final_pipeline = ColumnTransformer([
            ("cat", cat_pipeline, cat_attrs),
            ("num", num_pipeline, num_attrs)
        ])
    
    prepared_data = final_pipeline.fit_transform(X_train)

    X_train = pd.DataFrame(prepared_data, columns=final_attrs)

    return X_train, y_train



