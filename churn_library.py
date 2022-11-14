"""
Library for identify credit card customers that are most likely to churn

Author: Yue Chang

Date: 14 Nov 2022
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()


def import_data(pth: str) -> pd.DataFrame:
    """Returns dataframe for the csv found at pth

    Args:
        pth (str): a path to the csv

    Returns:
        pd.DataFrame: pandas dataframe
    """
    df = pd.read_csv(pth)
    print('Data sucessfully imported.')
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """perform eda on df and save figures to images folder

    Args:
        df (pd.DataFrame): pandas dataframe
    """
    print("Start performing EDA. ")
    # create folders for plots
    if not os.path.exists('images/eda'):
        os.mkdir('images/eda')

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('images/eda/churn_distribution.png')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_distribution.png')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_distribution.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/total_transaction_distribution.png')

    plt.figure(figsize=(20, 20))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/heatmap.png')
    print("Successfully performed EDA")


def encoder_helper(
        df: pd.DataFrame,
        category_list: list,
        response: str) -> pd.DataFrame:
    """Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    Args:
        df (pd.DataFrame): pandas dataframe
        category_list (list): list of columns that contain categorical features
        response (str): string of response name [optional argument that could be used for
        naming variables or index y column]

    Returns:
        pd.DataFrame: pandas dataframe with new encoded columns
    """
    # Reference:
    # https://stackoverflow.com/questions/73583250/target-encoding-multiple-columns-in-pandas-python
    temp_df = df[category_list].apply(
        lambda s: s.map(df[response].groupby(s).mean()))
    df = df.join(temp_df, rsuffix=f'_{response}')
    return df


def perform_feature_engineering(df: pd.DataFrame, response: str) -> tuple:
    """Perform feature engineering on dataframe

    Args:
        df (pd.DataFrame): pandas dataframe
        response (str): string of response name [optional argument that could be used for naming
        variables or index y column]

    Returns:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    print("Start Perform feature engineering. ")
    y = df[response]
    X = pd.DataFrame()
    df = encoder_helper(df, [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ], response)
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    # This cell may take up to 15-20 minutes to run
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print('Successfully performed feature engineering')
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train: list,
                                y_test: list,
                                y_train_preds_lr: list,
                                y_train_preds_rf: list,
                                y_test_preds_lr: list,
                                y_test_preds_rf: list) -> None:
    """Produces classification report for training and testing results and stores report as image
    in images folder

    Args:
        y_train (list): training response values
        y_test (list): test response values
        y_train_preds_lr (list): training predictions from logistic regression
        y_train_preds_rf (list): training predictions from random forest
        y_test_preds_lr (list): test predictions from logistic regression
        y_test_preds_rf (list): test predictions from random forest
    """
    if not os.path.exists('images/results'):
        os.mkdir('images/results')

    plt.figure()
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/lr_results.png')

    plt.figure()
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/rf_results.png')


def feature_importance_plot(
        model: object,
        X_data: pd.DataFrame,
        output_pth: str) -> None:
    """Creates and stores the feature importances in pth

    Args:
        model (object): model object containing feature_importances_
        X_data (pd.DataFrame): pandas dataframe of X values
        output_pth (str): path to store the figure
    """
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(f'{output_pth}/feature_importance.png')


def train_models(
        X_train: list,
        X_test: list,
        y_train: list,
        y_test: list) -> None:
    """Train, store model results: images + scores, and store models

    Args:
        X_train (list): X training data
        X_test (list): X testing data
        y_train (list): y training data
        y_test (list): y testing data
    """
    print("Start training models")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    print("Stored models scores")

    # plot roc curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/results/roc_curve_results.png')
    print("Stored ROC curve plot")

    # feature importance plot
    feature_importance_plot(cv_rfc.best_estimator_, X_train, 'images/results/')
    print("Stored feature importance plot")

    # save best model
    if not os.path.exists('models'):
        os.mkdir('models')
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    print("Stored best models")


if __name__ == "__main__":
    bank_df = import_data('data/bank_data.csv')
    perform_eda(bank_df)
    Xtrain, Xtest, ytrain, ytest = perform_feature_engineering(
        bank_df, 'Churn')
    train_models(Xtrain, Xtest, ytrain, ytest)
