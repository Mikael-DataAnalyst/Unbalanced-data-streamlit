import streamlit as st
import pandas as pd
from PIL import Image
import os
import json


st.write("test1qsdfsqdf")
st.write("test1qsdfsqdf")

# Chargement des données

# Images
pouce_vert = Image.open('images/pouce_vert.png')
pouce_rouge = Image.open("images/pouce_rouge.png")
logo = Image.open("images/logo.jpg")

# Données

data_test = pd.read_parquet("data/small_test.parquet")
with open("lgbm_model1.pkl", 'rb') as file:
    model = pickle.load(file)
col_info = pd.read_csv("data/col_info.csv")
with open("shap_values.pkl", 'rb') as file :
    shap_values = pickle.load(file)
with open("data/dict_nn.txt") as file :
    tmp = file.read()
dict_nn = json.loads(tmp)


# set variables
expected_value = 0.07576103670792426
best_tresh_scoring1 = 0.03807862092076406 
best_tresh_scoring2 = 0.07268097663014822
probs = model.predict_proba(data_test)
pred_score1 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring1
pred_score2 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring2
feats = data_test.columns


data_test = data_test.set_index("SK_ID_CURR")
#data_test = data_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data_test = data_test.drop(data_test.columns[[0]], axis=1)
st.image(logo)