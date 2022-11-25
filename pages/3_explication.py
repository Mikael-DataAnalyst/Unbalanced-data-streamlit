import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import shap
import numpy as np
import matplotlib.pyplot as plt

from credit_app import client_list, shap_values,id_client, expected_value, data_test


st.write('''
    # Pourquoi le crédit a été refusé ou accepté
    Les données en bleu baisse le risque de non remboursement

    Les données en rouge l'augmente
    ''')

idx = st.session_state["idx"]




fig = shap.force_plot(expected_value, shap_values[idx],
                        data_test.iloc[idx].round(2),matplotlib=True,
                        show = False,
                        contribution_threshold = 0.01,
                        text_rotation = 45)
st.pyplot(fig,bbox_inches='tight')
plt.clf()

'''data_client = data_test.iloc[idx,main_feat_user].round(2)
data_client["decision_influence"]=shap_values[idx,main_feat_user]
st.dataframe(data_client)'''


shap_user = shap_values[idx]
shap_user_importance = np.argsort(shap_user)
top_user_n = st.slider(
    "Combien de Données positives et négatives voulez vous voir",
    1,10,5 
)
neg_indexes = shap_user_importance[:top_user_n].tolist()
pos_indexes = shap_user_importance[-top_user_n:].tolist()
main_feat_user = neg_indexes + pos_indexes
main_feat_name = data_test.columns[main_feat_user]

fig = shap.force_plot(expected_value, shap_values[idx,main_feat_user],
                    data_test.iloc[idx,main_feat_user].round(2),
                    matplotlib=True,show=False,
                    figsize=(20,5),
                    contribution_threshold = -1,
                    text_rotation = 45)
st.pyplot(fig,bbox_inches='tight')
plt.clf()

fig = shap.decision_plot(expected_value, shap_values[idx], data_test.iloc[idx])
st.pyplot(fig)
plt.clf()