import pandas as pd
import numpy as np
import re
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score, confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve, make_scorer, roc_curve
from sklearn.neighbors import BallTree
import shap
from contextlib import contextmanager
import time
import json


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def load_model():
    with open("saved_data/model1.pkl", 'rb') as file:
        model = pickle.load(file)
        return model

def load_df():
    new_train = pd.read_parquet("saved_data/small_train1.parquet")
    new_train = new_train.drop(new_train.columns[0], axis=1)
    new_test = pd.read_parquet("saved_data/small_test1.parquet")
    new_test = new_test.drop(new_test.columns[0], axis=1)
    new_train = new_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    new_test = new_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    data_test = new_test.set_index("SK_ID_CURR").drop(columns = "TARGET")
    return new_train, data_test

def get_data_train(new_train):
    data = new_train.drop(["TARGET"], axis = 1)
    data = data.set_index(['SK_ID_CURR'])
    data = data.replace([np.inf, -np.inf], 0)
    y = new_train["TARGET"]
    feats = data.columns
    return data, y, feats

def get_scoring(new_train):
    annuity_mean = new_train["AMT_ANNUITY"].mean()
    loan_mean = new_train["AMT_CREDIT"].mean()
    return annuity_mean, loan_mean



def results(y, y_pred,new_train):
    print('Accuracy score for Testing Dataset = ', accuracy_score(y_pred, y))
    print('Roc auc score for Testing Dataset = ', roc_auc_score(y_pred, y))
    annuity_mean, loan_mean = get_scoring(new_train)
    cf = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cf).plot()
    
    tn, fp, fn, tp = cf.ravel()
    loose = fp * annuity_mean
    benefit = tn * annuity_mean - fn *loan_mean
    save_money = tp * loan_mean

    print('Bon crédit accordé:', tn)

def get_threshold(data,y,model,new_train):
    train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)
    
    y_pred = model.predict(valid_x)
    probs = model.predict_proba(valid_x)
    annuity_mean, loan_mean = get_scoring(new_train)
    
    cf = confusion_matrix(valid_y, y_pred)
    ConfusionMatrixDisplay(cf).plot()
    

    probs_1d = probs[:,1]
    fpr, tpr, thresholds = roc_curve(valid_y, probs_1d)
    gmeans = np.sqrt(tpr * (1-fpr))
    positive = len(valid_y[valid_y == 1])
    negative = len(valid_y[valid_y == 0])
    #Make the more money
    scoring1 = ((1-fpr)*annuity_mean*negative) - ((1-tpr)*loan_mean*positive)
    #Make money and save the more customers
    scoring2 = ((1-fpr)*annuity_mean*negative) - ((1-tpr)*loan_mean*positive) - (fpr * annuity_mean*negative)
    # best balance profit / loose
    ix = np.argmax(scoring1)
    print('Best Threshold=%f, scoring=%.3f' % (thresholds[ix], scoring1[ix]))
    test_pred_tresh = (model.predict_proba(valid_x)[:,1] >= thresholds[ix]).astype(int) 
    results(valid_y, test_pred_tresh,new_train)
    best_tresh_scoring1 = thresholds[ix]
    # best balance customers / Benefit
    ix = np.argmax(scoring2)
    print('Best Threshold=%f, scoring=%.3f' % (thresholds[ix], scoring2[ix]))
    test_pred_tresh = (model.predict_proba(valid_x)[:,1] >= thresholds[ix]).astype(int) 
    results(valid_y, test_pred_tresh,new_train)
    best_tresh_scoring2 = thresholds[ix]
    return best_tresh_scoring1, best_tresh_scoring2


def get_shap_values(model, data_test, data ,y):
    train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True, stratify=y, random_state=1301)
    explainer = shap.TreeExplainer(model,data = train_x, model_output = "probability", feature_perturbation="interventional") 
    shap_values = explainer.shap_values(data_test)
    # Save shap_values
    pkl_filename = "saved_data/shap_values1.pkl"
    with open(pkl_filename, 'wb') as file :
        pickle.dump(shap_values, file)
    return shap_values


def get_dict_nn(data_test,new_train,model, shap_values):
    data, y, feats = get_data_train(new_train)
    best_tresh_scoring1, best_tresh_scoring2 = get_threshold(data,y, model,new_train)
    client_list = data_test.index.to_list()
    probs = model.predict_proba(data_test)
    probs_df =pd.DataFrame(probs)
    clients= pd.DataFrame(data_test.index)    
    pred_score1 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring1
    pred_score2 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring2
    clients["pred_score_1"] = pred_score1
    clients["pred_score_2"] = pred_score2
    clients["prob_1"] = probs[:,1]

    tree = BallTree(probs, leaf_size = 2)
    dist, nn_prob = tree.query(probs, k=10)
    nn_prob_list = nn_prob.tolist()
    tree = BallTree(shap_values, leaf_size = 2)
    dist, nn_shap_values = tree.query(shap_values, k=10)
    nn_shap_values_list = nn_shap_values.tolist()
    dict_nn = {}
    for client in clients["SK_ID_CURR"] :
        idx = client_list.index(client)
        dict_nn[client]= [nn_prob_list[idx],nn_shap_values_list[idx]]
    with open('saved_data/dict_nn1.txt', 'w') as convert_file:
        convert_file.write(json.dumps(dict_nn))

def column_info(new_train):
    columns_df = new_train.columns.tolist()
    col_info = pd.read_csv("/Users/mikae/OneDrive/Documents/Formation/Data Scientist/Projet_7/data/HomeCredit_columns_description.csv",encoding = "ISO-8859-1")
    col_info = col_info[col_info["Table"]=="application_{train|test}.csv"]
    col_info = col_info[col_info["Row"].isin(columns_df)]
    col_info = col_info[["Row","Description"]]
    col_info = col_info.set_index("Row")
    col_info.to_csv("saved_data/col_info.csv")

def main():
    model = load_model()
    new_train, data_test = load_df()
    data, y, feats = get_data_train(new_train)
    annuity_mean, loan_mean = get_scoring(new_train)
    with timer("Shap values"):
        shap_values = get_shap_values(model, data_test, data,y)
    with timer("Dictionnary for nearest neighboors"):
        get_dict_nn(data_test, new_train,model, shap_values)
    column_info(new_train)

if __name__ == "__main__":
    with timer("Full model run"):
        main()