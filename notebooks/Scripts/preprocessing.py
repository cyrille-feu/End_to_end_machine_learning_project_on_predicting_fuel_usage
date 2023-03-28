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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


import warnings
warnings.filterwarnings('ignore')


class CustomColTransfomer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X, y=None):
        if "Origin" in X.columns:
            X["Origin"] = X["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
        else:
            X["Model Year"] = X["Model Year"].astype(str)
        return X

def num_pipeline_transformer():
    '''
    Function to process numerical transformations
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object

    '''

    num_attrs = ['Cylinders','Displacement','Horsepower','Weight','Acceleration']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median"))
    ])

    return num_attrs, num_pipeline

def cat_pipeline_transformer():
    '''
    Function to process categorical transformations
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object

    '''

    low_cat_attrs = ["Origin"]
    high_cat_attrs = ['Model Year']

    cat_pipeline_low = Pipeline ([
        ('col_map', CustomColTransfomer()),
        ('encoder_low', OneHotEncoder())
        ])

    cat_pipeline_high = Pipeline ([
        ('col_map', CustomColTransfomer()),
        ('encoder_high', OrdinalEncoder())
        ])

    return low_cat_attrs, cat_pipeline_low, high_cat_attrs, cat_pipeline_high


def preprocess(df, train=False, scaler=None):
    '''
    Function to process an entire dataset
    Returns:
        X_: features 
        y_: label 

    '''

    X_ = df.drop("MPG", axis=1)
    y_ = df["MPG"].copy()

    num_attrs, num_pipeline = num_pipeline_transformer()
    low_cat_attrs, cat_pipeline_low, high_cat_attrs, cat_pipeline_high = cat_pipeline_transformer()

    final_attrs = ['Model Year']+['Germany', 'India', 'USA'] + num_attrs

    final_pipeline = ColumnTransformer([
        ("cat_high", cat_pipeline_high, high_cat_attrs),
        ("cat_low", cat_pipeline_low, low_cat_attrs),
        ("num", num_pipeline, num_attrs)
    ])
    prepared_data = pd.DataFrame(final_pipeline.fit_transform(X_),columns=final_attrs)

    if train:
        std_scaler=StandardScaler()
        scaled_part = pd.DataFrame(std_scaler.fit_transform (prepared_data.iloc[:,4:]),columns=num_attrs)
    else:
        std_scaler=scaler
        scaled_part = pd.DataFrame(std_scaler.transform (prepared_data.iloc[:,4:]),columns=num_attrs)

    prepared_data[num_attrs]=scaled_part[num_attrs]

    return prepared_data, y_, std_scaler



