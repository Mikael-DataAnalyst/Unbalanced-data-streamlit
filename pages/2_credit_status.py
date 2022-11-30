import streamlit as st
import numpy as np

from credit_app import best_tresh_scoring1, best_tresh_scoring2, feats, model, data_test
from credit_app import pouce_rouge, pouce_vert, probs, selection_client
id_client = st.session_state["new_client"]

##############
# Select Box
##############

col1 = st.sidebar

changement = col1.checkbox("Voulez-vous changer de client")
if changement:    
    client_list = data_test.index.to_list()
    col1.header("Sélection du client:")
    selection = col1.selectbox(
            "Quel client ?",
            client_list
        )
    # save variables to use on other pages
    selection_client(selection)

idx = st.session_state["client_idx"]
col1.header('Client sélectionné')
col1.write("Client ID :",str(id_client))

option = col1.radio(
    "Quel scoring ?",
    ("Le plus rentable","Le plus de client")
)
if option == "Le plus de client" :
    col1.write("Vous avez choisi de favoriser le recrutement client")
    col1.write("Le seuil d'acceptation est de 7.3 %")
    threshold = best_tresh_scoring2
if option == "Le plus rentable" :
    col1.write("Vous avez choisi la sécurité financière")
    col1.write("Le seuil d'acceptation est de 3.8 %")
    threshold = best_tresh_scoring1

# Accord ou non du crédit en fonction de l'option
st.write("""
## Etat de la demande de crédit
""")
st.write("""
    ***
    """)

col1, col2, col3 = st.columns(3)
if probs[idx][1]<=threshold:
    col1.image(pouce_vert, width=150)
    col2.subheader("Le crédit est accordé")
else :
    col1.image(pouce_rouge, width=150)
    col2.subheader("Le crédit n'est pas accordé")
# Predict with trained model
st.write("""
## Prédictions de remboursement
""")
proba_remboursement = str(round(probs[idx][0]*100,2))
st.write(
    "Probabilité de remboursement du prêt : ", 
    proba_remboursement,"%")

proba_non_remboursement = str(round(probs[idx][1]*100,2))
st.write(
    f"Probabilité de non remboursement du prêt : *{proba_non_remboursement}*%")