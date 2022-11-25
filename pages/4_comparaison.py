import streamlit as st
import json
import shap
from credit_app import shap_values, data_test, expected_value,clients

with open("data/dict_nn.txt") as file :
    tmp = file.read()
dict_nn = json.loads(tmp)
id_client = st.session_state["id_client"]

idx = st.session_state["idx"]
id_nn_prob = dict_nn[str(id_client)][0]
id_nn_shap = dict_nn[str(id_client)][1]


st.header("Données globales")
shap.summary_plot(shap_values, data_test, show = False, max_display = 15, plot_size = (10,5))
st.pyplot( bbox_inches='tight')
plt.clf()

st.header('Comparaison avec un groupe de client similaires')
option = st.selectbox(
    "Quelles similarités ?",
    ("Probabilité de remboursement","Valeures influencantes")
)
if option == "Probabilité de remboursement" :
    id_nn = id_nn_prob
if option == "Valeures influencantes" :
    id_nn = id_nn_shap

col2, col3 = st.columns([2,1])

col3.dataframe(clients.iloc[id_nn])
r = shap.decision_plot(expected_value, shap_values[id_nn], data_test.iloc[id_nn],
    highlight=0,
    feature_display_range=slice(None, -11, -1),
    return_objects = True)
col2.pyplot()

shap.decision_plot(expected_value, shap_values[idx], data_test.iloc[idx],
                feature_order=r.feature_idx, xlim=r.xlim,
                feature_display_range=slice(None, -11, -1))
col2.pyplot()


shap_group = shap_values[id_nn]
shap_group_importance = np.argsort(shap_group).tolist()
tmp = []
for x in shap_group_importance:
    tmp = tmp+x[:2]
    tmp = tmp+x[-2:]
most_importance_feats_group = list(set(tmp))

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
st_shap(shap.force_plot(expected_value,
            shap_values[id_nn,:][:,most_importance_feats_group],
            data_test.iloc[id_nn,most_importance_feats_group]),
            400)