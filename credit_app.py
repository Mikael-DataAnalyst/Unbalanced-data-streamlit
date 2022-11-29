import streamlit as st
import pandas as pd
from PIL import Image
import pickle

import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit application",
    page_icon="moneybag",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Application développée par Mikael André"
    }
)


##########################
# Chargement des données #
##########################
# Images
@st.cache
def load_image():
    pouce_vert = Image.open('images/pouce_vert.png')
    pouce_rouge = Image.open("images/pouce_rouge.png")
    logo = Image.open("images/logo.jpg")
    return pouce_vert, pouce_rouge, logo

pouce_vert, pouce_rouge, logo = load_image()

# Données
def load_data():
    data_test = pd.read_parquet("data/small_test.parquet")
    data_test = data_test.set_index("SK_ID_CURR")
    #data_test = data_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    data_test = data_test.drop(data_test.columns[[0]], axis=1)
    with open("data/model.pkl", 'rb') as file:
        model = pickle.load(file)
    col_info = pd.read_csv("data/col_info.csv")
    with open("data/shap_values.pkl", 'rb') as file :
        shap_values = pickle.load(file)
    info_client = pd.read_parquet("data/info_client.parquet")
    return data_test, model, col_info, shap_values, info_client

data_test, model, col_info, shap_values, info_client = load_data()

# set variables
expected_value = 0.07576103670792426
best_tresh_scoring1 = 0.03807862092076406 
best_tresh_scoring2 = 0.07268097663014822
probs = model.predict_proba(data_test)
pred_score1 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring1
pred_score2 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring2
feats = data_test.columns
client_list = data_test.index.to_list()
st.session_state["id_client"] = 100001
st.session_state["idx"] = client_list.index(100001)
col1 = st.sidebar
col1.image(logo)


# Fonctions
def glossaire():
    features_list = col_info["features"]

    expander1 = st.expander("Voir Glossaire")
    explication = expander1.multiselect("Quel terme", options = features_list, help="Tapez votre recherche")
    return expander1.table(col_info[col_info["features"].isin(explication)][["features","Description"]])

#st.sidebar.image(logo, width=150)
col1 = st.sidebar
client_list = data_test.index.to_list()
col1.header("Sélection du client:")
selection = col1.selectbox(
        "Quel client ?",
        client_list
    )
# save variables to use on other pages
st.session_state["id_client"] = selection
st.session_state["idx"] = client_list.index(selection)

# summary plot
st.header("Données globales")
st.write("Données qui influençent le plus la décision")

def summary_plot(shap_values, data_test):
    fig = shap.summary_plot(shap_values, data_test, show = False, max_display = 15, plot_size = (10,5))
    return fig
fig = summary_plot(shap_values, data_test)

st.pyplot(fig,bbox_inches='tight')
plt.clf()


