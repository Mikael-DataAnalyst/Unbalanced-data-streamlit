# Application pour crédit
***
## Application pour l'entreprise "Prêt à Dépenser"
Permettre au chargé de relation client d'expliquer de façon transparente les décisions d'octroi de crédit.
Permettre aux clients de disposer de leurs informations personnelles et de les explorer facilement.

Source des données : [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)  
Source pour le preprocessing : [Aguiar](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features)
### Modèle

Deux branches :
    - main sans utiliser smote
    - sampling avec un over sampling et un under sampling

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
[Sans sampling](https://mikael-dataanalyst-pret-a-depenser-home-7jlehp.streamlit.app/)  
[Avec sampling](https://mikael-dataanalyst-pret-a-depenser-home-sampling-5nai7s.streamlit.app/)
