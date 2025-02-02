import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import seaborn as sb
import matplotlib.pyplot as plt

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data
def factorial_analysys():
    data = pd.read_csv('dataIN/MortalitateEU.csv')
    numeric_data = data.iloc[:, 2:]
    cleaned_data = clean_data(numeric_data)

    #Standartizare
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)

    # Modelul factorial
    nr_var = normalized_data.shape[1]
    modelFact = FactorAnalyzer(n_factors=nr_var, rotation=None)
    F = modelFact.fit(normalized_data)

    scoruri = modelFact.transform(normalized_data)
    scoruri_df = pd.DataFrame(scoruri, columns=['F'+str(i) for i in range(scoruri.shape[1])])
    scoruri_df.to_csv('dataOUT/scoruri.csv')
    print(scoruri_df.head())

    #Calcul varianta
    # [0] ca sa luam cata varianta explica fiecare factor
    varianta = modelFact.get_factor_variance()[0]
    varianta_df = pd.DataFrame(varianta, index=numeric_data.columns, columns=['Variance'])
    varianta_df.to_csv('dataOUT/varianta_fiecare_factor.csv')
    print(varianta_df)

    # Corelatii
    corr = modelFact.loadings_
    corr_df = pd.DataFrame(corr, index=numeric_data.columns, columns=['F'+str(i) for i in range(scoruri.shape[1])])
    corr_df.to_csv('dataOUT/Corelatii.csv')
    print(corr_df)
    # Corelograma
    sb.heatmap(corr_df)
    plt.show()

    # Comunalitati
    comm = modelFact.get_communalities()
    comm_df = pd.DataFrame(comm, index=numeric_data.columns)
    comm_df.to_csv('dataOUT/comunalitatii.csv')

    # Tests
    # KMO
    kmo_results = calculate_kmo(normalized_data)
    print('KMO individual[0]: ', kmo_results[0])
    print('KMO general[1]: ', kmo_results[1])

    # Barlett
    barlett_test = calculate_bartlett_sphericity(normalized_data)
    print('P-value: ', barlett_test[1])

    # Trasare plot scoruri
    plt.scatter(scoruri_df['F1'], scoruri_df['F2'])
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.show()



factorial_analysys()
