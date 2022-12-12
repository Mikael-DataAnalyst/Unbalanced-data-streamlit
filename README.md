# Projet_7
***
## Application pour l'entreprise "Prêt à Dépenser"
Permettre au chargé de relation client d'expliquer de façon transparente les décisions d'octroi de crédit.
Permettre aux clients de disposer de leurs informations personnelles et de les explorer facilement.

### Modèle
code_model.py permet de faire le préprocessing et le train du modèle
Permet de sauvegarder :  
    - le model  
    - le data test modifié  
    - les infos clients brutes  

### Génération des fichiers pour l'API
code_data.py permet de générer les données pour l'API :  
    - shap_values  
    - le glossaire  
    - dictionnaire de variables  
    - dictionnaire de nearest neighboors  

### L'application est disponible sur Streamlit Cloud :
https://mikael-dataanalyst-pret-a-depenser-home-7jlehp.streamlit.app/