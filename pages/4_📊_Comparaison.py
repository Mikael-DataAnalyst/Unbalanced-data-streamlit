import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import json
import shap
import matplotlib.pyplot as plt
import numpy as np
from Home import shap_values, data_test, expected_value, client_list
from Home import pred_score1, pred_score2, probs, glossaire

st.set_page_config(
    page_title="Explication de la décision",
    page_icon="📊",layout="wide")

with open("data_saved/dict_nn1.txt") as file :
        tmp = file.read()
dict_nn = json.loads(tmp)

id_client = st.session_state["new_client"]
idx_nn_prob = dict_nn[str(id_client)][0]
idx_nn_shap = dict_nn[str(id_client)][1]
st.session_state["idx_nn_prob"]= idx_nn_prob
st.session_state["idx_nn_shap"]= idx_nn_shap


clients= pd.DataFrame(data_test.index)
clients["pred_score_1"] = pred_score1
clients["pred_score_2"] = pred_score2
clients["prob_1"] = probs[:,1]

col1 = st.sidebar
col1.header('Client sélectionné')
col1.write("Client ID :",str(id_client))

st.header('Comparaison avec un groupe de client similaires')
option = col1.selectbox(
    "Quelles similarités ?",
    ("Probabilité de remboursement","Valeures influencantes")
)
if option == "Probabilité de remboursement" :
    id_nn = idx_nn_prob
if option == "Valeures influencantes" :
    id_nn = idx_nn_shap



col2, col3 = st.columns([2,1])

col3.dataframe(clients.iloc[id_nn])
r = shap.decision_plot(expected_value, shap_values[id_nn], data_test.iloc[id_nn],
    highlight=0,
    feature_display_range=slice(None, -11, -1),
    return_objects = True)
col2.pyplot()

def shap_group(id_nn, n_features = 2):
    shap_group = shap_values[id_nn]
    shap_group_importance = np.argsort(shap_group).tolist()
    tmp = []
    for x in shap_group_importance:
        tmp = tmp+x[:n_features]
        tmp = tmp+x[-n_features:]
    most_importance_feats_group = list(set(tmp))
    return most_importance_feats_group

def st_shap(id_nn, height=400):
    plot = shap.force_plot(expected_value,
            shap_values[id_nn,:][:,shap_group(id_nn)],
            data_test.iloc[id_nn,shap_group(id_nn)])
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def st_shap2(id_nn, height=400, n_features=2):
    def shap_group(id_nn, n_features):
        shap_group = shap_values[id_nn]
        shap_group_importance = np.argsort(shap_group).tolist()
        tmp = []
        for x in shap_group_importance:
            tmp = tmp+x[:n_features]
            tmp = tmp+x[-n_features:]
        most_importance_feats_group = list(set(tmp))
        return most_importance_feats_group

    plot = shap.force_plot(expected_value,
            shap_values[id_nn,:][:,shap_group(id_nn, n_features)],
            data_test.iloc[id_nn,shap_group(id_nn, n_features)])

    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st_shap2(id_nn)
    

st.subheader("Comparer deux clients de ce groupe")
col2, col3 = st.columns([1,1])
id_1 = col2.selectbox("Client 1",clients.iloc[id_nn])
def decision_plot1(id):
    idx = client_list.index(id)
    fig = shap.decision_plot(expected_value, shap_values[idx], data_test.iloc[idx],
                feature_order=r.feature_idx, xlim=r.xlim,
                feature_display_range=slice(None, -11, -1))
    return fig
fig = decision_plot1(id_1)
col2.pyplot(fig)

id_2 = col3.selectbox("Client 2",clients.iloc[id_nn] )
fig = decision_plot1(id_2)
col3.pyplot(fig)

def get_idx(id):
    idx = client_list.index(id)
    return idx

st_shap2([get_idx(id_1),get_idx(id_2),get_idx(id_1)], n_features=5)

glossaire()
