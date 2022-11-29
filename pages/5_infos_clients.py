import pandas as pd
import streamlit as st

from credit_app import col_info

col_info

features_list = col_info["Row"]
descriptions = col_info["Description"]

expander1 = st.expander("Voir Glossaire")
explication = expander1.multiselect("Glossaire", options = features_list, help="Quel définition")

st.write(explication[0], " :", col_info[col_info["Row"]== explication[0]]["Description"])
if len(explication) !=0:
    for i in range(len(explication)):
        expander1.write(col_info[col_info["Row"]== explication[i]])
        #expander1.write(explication[0], " :", col_info[col_info["Row"]== explication[0]]["Description"])