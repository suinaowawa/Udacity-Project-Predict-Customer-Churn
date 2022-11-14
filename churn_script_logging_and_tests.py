"""
Unit tests and logging for churn libraray

Author: Yue Chang

Date: 14 Nov 2022
"""
import os
import logging

import pandas as pd

import churn_library as churn_lib


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = churn_lib.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        df = churn_lib.import_data('./data/bank_data.csv')
        churn_lib.perform_eda(df)
        assert os.path.exists('images/eda')
        assert len(os.listdir('images/eda')) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: Images not created correctly.")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        df = churn_lib.import_data('./data/bank_data.csv')
        category_list = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        new_df = churn_lib.encoder_helper(df, category_list, 'Churn')
        assert isinstance(new_df, pd.DataFrame)
        assert len(new_df.columns) > len(df.columns)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Failed to return dataframe with new encoded columns")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        df = churn_lib.import_data('./data/bank_data.csv')
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        res = churn_lib.perform_feature_engineering(df, 'Churn')
        assert res is not None
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Feature Enginnering process failed. ")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        df = churn_lib.import_data('./data/bank_data.csv')
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        res = churn_lib.perform_feature_engineering(df, 'Churn')
        churn_lib.train_models(*res)
        assert os.path.exists('images/results')
        assert len(os.listdir('images/results')) > 0
        assert os.path.exists('models')
        assert len(os.listdir('models')) > 0
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: models or result images not created correctly.")
        raise err
