import pandas as pd
import streamlit as st

from credit_app import glossaire, info_client, client_list

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

id_client = st.session_state["client_id"]
idx = st.session_state["client_idx"]
idx_nn_prob = st.session_state["idx_nn_prob"]
idx_nn_shap = st.session_state["idx_nn_shap"]

df_client = info_client[info_client["SK_ID_CURR"]== id_client].T

col1 = st.sidebar
col1.header('Client sélectionné')
col1.write("Client ID :",str(id_client))

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = col1.checkbox("Add filters")

    if not modify:
        return df.head(50)

    df = df.copy()

    modification_container = st.container()

    with modification_container:
        
        unique_client = col1.checkbox("Client actuel")
        if unique_client:
            df = df[df["SK_ID_CURR"]==id_client]

        group_nn = col1.checkbox("Clients comparables")
        if group_nn:
            left, right = st.columns((1, 20))
            
            to_add_nn = right.selectbox(
            "Quel groupe de client ?",
            ("Probabilité de remboursement","Valeures influencantes")
        )
            if to_add_nn == "Probabilité de remboursement" :
                idx_nn = idx_nn_prob
            if to_add_nn == "Valeures influencantes" :
                idx_nn = idx_nn_shap
            df = info_client.iloc[idx_nn]

        multiple_client = col1.checkbox("Plus de clients")
        if multiple_client:
            left, right = st.columns((1, 20))

            to_filter_clients = right.multiselect("Sélection des clients supplémentaires", info_client["SK_ID_CURR"])

            df = df.append(info_client[info_client["SK_ID_CURR"].isin(to_filter_clients)])

        column_selection = col1.checkbox("Sélection des données")
        if column_selection:
            left, right = st.columns((1, 20))
            to_select_column = st.multiselect("Quelles données afficher ?", df.columns.drop(["SK_ID_CURR"]))
            to_select_column =["SK_ID_CURR"] + to_select_column
            df = df[to_select_column]

        filter_column = col1.checkbox("Filtrer les données")
        if filter_column:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns.drop(["SK_ID_CURR"]))
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

    


st.dataframe(filter_dataframe(info_client))
glossaire()
