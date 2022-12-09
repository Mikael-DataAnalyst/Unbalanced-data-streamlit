import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import json
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
@st.experimental_singleton
def load_image():
    pouce_vert = Image.open('images/pouce_vert.png')
    pouce_rouge = Image.open("images/pouce_rouge.png")
    logo = Image.open("images/logo.jpg")
    return pouce_vert, pouce_rouge, logo

pouce_vert, pouce_rouge, logo = load_image()

# Données
@st.experimental_memo
def load_data():
    data_test = pd.read_parquet("saved_data/small_test1.parquet")
    data_test = data_test.set_index("SK_ID_CURR")
    #data_test = data_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    data_test = data_test.drop(data_test.columns[[0,1]], axis=1)
    with open("saved_data/model1.pkl", 'rb') as file:
        model = pickle.load(file)
    col_info = pd.read_csv("saved_data/col_info.csv")
    with open("saved_data/shap_values1.pkl", 'rb') as file :
        shap_values = pickle.load(file)
    info_client = pd.read_parquet("saved_data/info_client.parquet")
    with open("saved_data/dict_nn1.txt") as file :
        tmp = file.read()
    dict_nn = json.loads(tmp)
    return data_test, model, col_info, shap_values, info_client, dict_nn

data_test, model, col_info, shap_values, info_client, dict_nn = load_data()

# set variables
expected_value = 0.07576103670792426
best_tresh_scoring1 = 0.03807862092076406 
best_tresh_scoring2 = 0.07268097663014822
probs = model.predict_proba(data_test)
pred_score1 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring1
pred_score2 = model.predict_proba(data_test)[:,1] >= best_tresh_scoring2
feats = data_test.columns
client_list = data_test.index.to_list()
client_list = data_test.index.to_list()

col1 = st.sidebar


col1.header("Sélection du client:")

selection = col1.selectbox(
        "Quel client ?",
        client_list, key="client_id",)

check_key = st.session_state.get("new_client", "empty")


@st.experimental_singleton
def selection_client(selection):
    st.session_state["new_client"] = selection
    st.session_state["client_idx"] =client_list.index(selection)
    idx_nn_prob = dict_nn[str(st.session_state["new_client"])][0]
    idx_nn_shap = dict_nn[str(st.session_state["new_client"])][1]
    st.session_state["idx_nn_prob"]= idx_nn_prob
    st.session_state["idx_nn_shap"]= idx_nn_shap    

selection_client(selection)
if check_key == "empty":
    col1.header('Client sélectionné')
    col1.write('Client ID : 100001')
else:
    col1.header('Client sélectionné')
    col1.write("Client ID :",str(st.session_state.new_client))
# save variables to use on other pages




idx = st.session_state["client_idx"]


col1.image(logo)


# Fonctions
def glossaire():
    features_list = col_info["features"]

    expander1 = st.expander("Voir Glossaire")
    explication = expander1.multiselect("Quel terme", options = features_list, help="Tapez votre recherche")
    return expander1.table(col_info[col_info["features"].isin(explication)][["features","Description"]])

st.write("# Bienvenue sur l'application pour les crédits ")
st.markdown(
    """ 
    Cette application vous permet de voir la décision d'octroi de crédit
     pour un client, de pouvoir lui expliquer cette décision 
     et de consulter ses informations descriptives.
    
    ### Quels clients ?
    La base de donnée est composée d'environ 50 000 clients en attente d'une décision.

    Plusieurs options vous seront proposées dans cette barre sur les différentes pages.
    ### Comment est basée la décision ?
    - Un algorythme prédit la probabilité de non remboursement du client
    - Vous pouvez choisir entre 2 choix :
        - 💰 Le plus rentable : le seuil acceptable est fixé à 3.8 %
            Nous nous assurons d'avoir le plus de client fiables mais refusont beaucoup de clients
        - 👫 Recrutement Client : le seuil acceptable est fixé à 7.3%
            Nous recrutons le maximum de clients tout en étant rentable

    #### Sommaire
    👈 Vous pouvez naviguer en sélectionnant les pages dans la barre de droite.
    - Statut du crédit :
        - Probabilité
        - Décision en fonction de la politique choisie
    - Explications :
        - Comments ses données influences sa probabilités
    - Comparaison :
        - Comparaison avec des clients similaires
    - Informations personnelles
        - Consultation des données brutes avant transformation
     """
)

# summary plot



# def summary_plot(shap_values, data_test):
#     shap.summary_plot(shap_values, data_test, show = False, max_display = 15, plot_size = (10,5))
    
# summary_plot(shap_values, data_test)

st.pyplot(bbox_inches='tight')
plt.clf()






