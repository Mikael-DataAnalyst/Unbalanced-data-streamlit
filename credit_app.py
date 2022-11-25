import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import json

##########################
# Chargement des données #
##########################
# Images
pouce_vert = Image.open('images/pouce_vert.png')
pouce_rouge = Image.open("images/pouce_rouge.png")
logo = Image.open("images/logo.jpg")
logo2 = Image.open("images/logo.jpg")
# Données

data_test = pd.read_parquet("data/small_test.parquet")
data_test = data_test.set_index("SK_ID_CURR")
#data_test = data_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data_test = data_test.drop(data_test.columns[[0]], axis=1)
with open("data/model.pkl", 'rb') as file:
    model = pickle.load(file)
col_info = pd.read_csv("data/col_info.csv")
with open("data/shap_values.pkl", 'rb') as file :
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
client_list = data_test.index.to_list()
id_client = 100001
a = 3
st.image(logo)
######################
# Input Text Box
######################
#st.sidebar.image(logo, width=150)
id_input = "Par exemple : 100013\n100001"

id_client = st.text_area("ID du client à renseigner", id_input)
id_client = id_client.splitlines()
id_client = id_client[1:] # Skips the sequence name (first line)
id_client = ''.join(id_client) # Concatenates list to string
id_client = int(id_client)
if id_client not in data_test.index.values:
    st.write("Ce client n'est pas dans la base de donnée")


## Selecting Client ID in a list
client_list = data_test.index.to_list()
list_choice = st.checkbox("Voulez-vous une liste des clients")
if list_choice:
    id_client = st.selectbox(
        "Quel client ?",
        client_list
    )
col1 = st.sidebar
## Selection scoring
col1.header("Clients sélectionneé:")
col1.write(id_client)

def return_client():
    return id_client

