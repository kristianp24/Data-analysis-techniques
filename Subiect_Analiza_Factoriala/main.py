from sklearn.preprocessing import StandardScaler
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity, FactorAnalyzer
import seaborn as sb
import pandas as pd
import numpy as np

def cerinta1():
    data_vod = pd.read_csv('dataIN/Vot.csv')
    coloane = ['Siruta', 'Localitate', 'Categorie']
    data = data_vod.apply(func=lambda x: x.iloc[2:].idxmin(), axis=1)
    cer2 = pd.concat([data_vod, data], axis=1)
    cer2.to_csv('dataOUT/cernta1.csv')

def cerinta2():
    data_vod = pd.read_csv('dataIN/Vot.csv')
    coduri_loc = pd.read_csv('dataIN/Coduri_localitati.csv')
    merged_data = data_vod.merge(coduri_loc, on='Siruta')
    columns_for_medie = data_vod.columns[3:]
    grouped_data = merged_data.groupby('Judet')[columns_for_medie].mean()
    grouped_data.to_csv('dataOUT/cerinta2.csv')


def clean_data(numeric_data:pd.DataFrame):
    if numeric_data.isna().any().any():
        for col in numeric_data.columns:
            if numeric_data[col].isna().any():
                numeric_data[col] = numeric_data[col].fillna(numeric_data[col].mean())

    return numeric_data
def analiza_factoriala():
    data_vod = pd.read_csv('dataIN/Vot.csv')
    cleaned_data = clean_data(data_vod.iloc[:,3:])
    print(cleaned_data)
    #Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)
    #print(normalized_data)
    df_normalized_data = pd.DataFrame(normalized_data)
    df_normalized_data = df_normalized_data.loc[:, df_normalized_data.var() > 0]

    # Scoruri
    nr_factors = min(df_normalized_data.shape[1], df_normalized_data.shape[0] - 1)
    model_Fact = FactorAnalyzer(n_factors=nr_factors, rotation=None)

    scoruri = model_Fact.fit_transform(df_normalized_data)
    print(scoruri)

    # Testul Barret
    test_barret = calculate_bartlett_sphericity(normalized_data)
    print('Valoarea p-value: ', test_barret[1])



analiza_factoriala()