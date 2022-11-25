import streamlit as st
import shap
import numpy as np

from credit_app import client_list, shap_values,id_client, expected_value, data_test

idx = client_list.index(id_client)
shap_user = shap_values[idx]
shap_user_importance = np.argsort(shap_user)
top_user_n = 5
neg_indexes = shap_user_importance[:5].tolist()
pos_indexes = shap_user_importance[-5:].tolist()
main_feat_user = neg_indexes + pos_indexes
main_feat_name = data_test.columns[main_feat_user]

fig = shap.force_plot(expected_value, shap_values[idx],
                        data_test.iloc[idx],matplotlib=True,
                        show = False,
                        contribution_threshold = 0.02,
                        text_rotation = 45)
st.pyplot(fig,bbox_inches='tight')