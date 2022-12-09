import glob
import os
import pandas as pd
import gc
import time
from nltk.tokenize import RegexpTokenizer
import numpy as np


# utilities
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import re
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# modeling 
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb

from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

import shap

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.max_columns = None

# Path for the files
path = "/Users/mikae/OneDrive/Documents/Formation/Data Scientist/Projet_7/data/"


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def nan_imputer(df):
    imputer_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer_most_frequent = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    for col in df.loc[:,df.columns != 'TARGET'] :
        if df[col].isna().sum() !=0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = imputer_mean.fit_transform(df[col].values.reshape(-1,1))
            if df[col].dtype == 'object':
                df[col] =imputer_most_frequent.fit_transform(df[col].values.reshape(-1,1))
    return df


def missing(df, threshold = 0.75):
    missing = (df.isnull().sum() / len(df)).sort_values(ascending = False)
    missing = missing.index[missing > threshold]
    if len(missing)>0 :
        print(len(missing), "column(s) contains more than", threshold*100, "% of Nan and were dropped")
    df = df.drop(columns = missing)
    return df
                
def application_train_test(num_rows = None, nan_as_category = True):
    # Read data and merge
    df = pd.read_csv(path+'application_train.csv', nrows= num_rows)
    test_df = pd.read_csv(path+'application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    df = missing(df)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    df = nan_imputer(df)

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    

    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
  
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = False, threshold = 0.75):
    bureau = pd.read_csv(path+'bureau.csv', nrows = num_rows)
    bb = pd.read_csv(path+'bureau_balance.csv', nrows = num_rows)
    print("bureau")
    bureau = missing(bureau)
    print("bureau_balance")
    bb = missing(bb)
   
    bureau = nan_imputer(bureau)
    bb = nan_imputer(bb)
    
    
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
   
    del bureau
    gc.collect()

    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = False):
    print("Previous application")
    prev = pd.read_csv(path+'previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= False)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    prev = missing(prev)
    prev = nan_imputer(prev)

        
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    del prev
    gc.collect()
    prev_agg = missing(prev_agg)
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = False):
    print("Pos cash balance")
    pos = pd.read_csv(path+'POS_CASH_balance.csv', nrows = num_rows)
    pos = missing(pos)
    pos = nan_imputer(pos)
        
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= False)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    pos_agg = missing(pos_agg)
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = False):
    print("installments payments")
    ins = pd.read_csv(path+'installments_payments.csv', nrows = num_rows)
    ins = missing(ins)
    ins = nan_imputer(ins)

    ins, cat_cols = one_hot_encoder(ins, nan_as_category= False)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    ins_agg = missing(ins_agg)
    ins_agg = ins_agg.fillna(0)   
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = False):
    print("credit_card_balance")
    cc = pd.read_csv(path+'credit_card_balance.csv', nrows = num_rows)
    cc = missing(cc)
    cc = nan_imputer(cc)
        
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= False)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    cc_agg = missing(cc_agg)
    return cc_agg

# Drop the highly correlated features
def drop_corr(df, threshold = 0.9):
    # Threshold for removing correlated variables

    # Absolute value correlation matrix
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(len(to_drop), 'columns with correlations above', threshold, "have been removed")
    df = df.drop(columns = to_drop)
    return df

# Separate train and test 
def train_test_separate(df):
    train_df = df[df['TARGET'].notnull()]
    train_df = train_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    test_df = df[df['TARGET'].isnull()]
    test_df = test_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
  
    return train_df, test_df

def column_info(train_df):
    columns_df = train_df.columns.tolist()
    col_info = pd.read_csv("HomeCredit_columns_description.csv",encoding = "ISO-8859-1")
    col_info = col_info[col_info["Table"]=="application_{train|test}.csv"]
    col_info = col_info[col_info["Row"].isin(columns_df)]
    col_info = col_info[["Row","Description"]]
    col_info = col_info.set_index("Row")
    col_info.to_csv("col_info.csv")


def get_data(train_df):
    train = train_df.set_index(['SK_ID_CURR'])
    train = train.drop(train_df.columns[0], axis=1)
    data = train.drop(["TARGET"], axis = 1)
    data = data.replace([np.inf, -np.inf], 0)
    y = train["TARGET"]
    return data, y

# Select only important features
def drop_feat(train_df,test_df, threshold=0.9):
    data, y = get_data(train_df)
    feats = data.columns
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(data.shape[1])
    model_lgbm = lgb.LGBMClassifier(objective='binary')
    # Create the model with several hyperparameters
    callbacks = [lgb.early_stopping(stopping_rounds = 200)]

    # Fit the model twice to avoid overfitting
    for i in range(2):
        
        # Split into training and validation set
        #Create train and validation set
        train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=i)
        
        # Train using early stopping
        model_lgbm.fit(train_x, train_y, callbacks = callbacks, eval_set = [(valid_x, valid_y)], 
                eval_metric = 'auc')
        
        # Record the feature importances
        feature_importances += model_lgbm.feature_importances_

    # Make sure to average feature importances! 
    feature_importances = feature_importances / 2
    feature_importances_df = pd.DataFrame({'feature': feats, 'importance': feature_importances}).sort_values('importance', ascending = False)
    feature_importances_df = feature_importances_df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    feature_importances_df['importance_normalized'] = feature_importances_df['importance'] / feature_importances_df['importance'].sum()
    feature_importances_df['cumulative_importance'] = np.cumsum(feature_importances_df['importance_normalized'])

    importance_index = np.min(np.where(feature_importances_df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    # Extract the features to drop
    features_to_drop = list(feature_importances_df[feature_importances_df['cumulative_importance'] > threshold]['feature'])
    print(len(features_to_drop))
    train_df = train_df.drop(columns = features_to_drop)
    test_df = test_df.drop(columns = features_to_drop)
    return train_df, test_df

def lgmb_model(train_df,test_df):
    data, y = get_data(train_df)
    #Create train and validation set
    train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)

    callbacks = [lgb.early_stopping(stopping_rounds = 100),
            #lgb.log_evaluation()
            ]
    fit_params={"eval_metric" : 'auc', 
                "eval_set" : [(valid_x,valid_y)],
                'eval_names': ['valid'],
                'callbacks' : callbacks,
                'categorical_feature': 'auto'}
    clf = LGBMClassifier(colsample_bytree=0.42665072497940254, max_depth=6, metric='None',
               min_child_samples=464, min_child_weight=100.0, n_estimators=5000,
               n_jobs=4, num_leaves=13, random_state=314, reg_alpha=0,
               reg_lambda=0, subsample=0.7008050755353639, scale_pos_weight=1)
    
    #Create train and validation set
    train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)

    clf.fit(train_x, train_y,**fit_params)

    # Save model
    pkl_filename = "model1.pkl"
    with open(pkl_filename, 'wb') as file :
        pickle.dump(clf, file)

    test = test_df.set_index(['SK_ID_CURR'])
    test = test.drop(train_df.columns[0], axis=1)
    data_test = test.drop(["TARGET"], axis = 1)
    data_test = data_test.replace([np.inf, -np.inf], 0)
    # Empty array for test predictions
    test_predictions = np.zeros(data_test.shape[0])

    test_predictions += clf.predict_proba(data_test, num_iteration=clf.best_iteration_)[:,1]
    submit = data_test.reset_index()[['SK_ID_CURR']]
    submit['TARGET'] = test_predictions
    submit.to_csv(submission_file_name, index = False)

    


def main(debug = False):
    num_rows = 20000 if debug else None
    df = application_train_test(num_rows)
    print("df shape:",df.shape, "Nan:",df.isna().sum().sum())
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        df.loc[:,df.columns != 'TARGET'] = df.fillna(0)
        del bureau
        gc.collect()
    print("After join, df shape:",df.shape, "Nan:", df.isna().sum().sum())
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        df.loc[:,df.columns != 'TARGET'] = df.fillna(0)
        del prev
        gc.collect()
    print("After join, df shape:",df.shape, "Nan:" ,df.isna().sum().sum())

    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        df.loc[:,df.columns != 'TARGET'] = df.fillna(0)

        del pos
        gc.collect()
    print("After join, df shape:",df.shape, "Nan:", df.isna().sum().sum())
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        df.loc[:,df.columns != 'TARGET'] = df.fillna(0)

        del ins
        gc.collect()
    print("After join, df shape:",df.shape, "Nan:", df.isna().sum().sum())
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        df.loc[:,df.columns != 'TARGET'] = df.fillna(0)

        del cc
        gc.collect()
    print("After join, df shape:",df.shape, "Nan:", df.isna().sum().sum())
    df = missing(df)
    with timer("Calculate correlations and drop"):
        df = drop_corr(df)
    print("df shape after drop highly correlated features:", df.shape, "Nan:", df.isna().sum().sum())
    with timer("Separate train and test"):
        train_df, test_df = train_test_separate(df)
        del df
        gc.collect()
    print("After separate, train_df shape : ", train_df.shape)
    print("After separate, test_df shape : ", test_df.shape)
    with timer("Info of columns"):
        column_info(train_df)
    with timer("Select only the most important features"):
        train_df, test_df = drop_feat(train_df, test_df)
    print("After drop, train_df shape : ", train_df.shape)
    print("After drop, test_df shape : ", test_df.shape)    
    with timer("Select only the most important features"):
        lgmb_model(train_df,test_df)

if __name__ == "__main__":
    submission_file_name = "submission_02.csv"
    with timer("Full model run"):
        main()

        