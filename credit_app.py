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
@st.cache
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
    return data_test, model, col_info, shap_values

data_test, model, col_info, shap_values = load_data()



# set variables
expected_value = 0.07576103670792426
best_tresh_scoring1 = 0.03807862092076406 
best_tresh_scoring2 = 0.07268097663014822
probs = model.predict_proba(data_test)
pred_score1 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring1
pred_score2 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring2
feats = data_test.columns
client_list = data_test.index.to_list()

col1 = st.sidebar
col1.image(logo)

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


# summary plot
st.header("Données globales")
st.write("Données qui influençent le plus la décision")
def summary_plot(shap_values, data_test):
    fig = shap.summary_plot(shap_values, data_test, show = False, max_display = 15, plot_size = (10,5))
    return fig
fig = summary_plot(shap_values, data_test)
st.pyplot(fig,bbox_inches='tight')
plt.clf()


