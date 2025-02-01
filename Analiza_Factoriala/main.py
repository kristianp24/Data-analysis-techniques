import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sb

def cerinta1():
    data = pd.read_csv('dataIN/CAEN2_2021_NSAL.csv', index_col='SIRUTA')
    cer1= data.apply(func= lambda x: np.round((x / x[0:].sum()) * 100, 2), axis= 0)
    cer1.to_csv('dataOUT/cerint1.csv')

def cerinta2():
    caen_data = pd.read_csv('dataIN/CAEN2_2021_NSAL.csv')
    pop_data = pd.read_csv('dataIN/PopulatieLocalitati.csv')

    merged_data = caen_data.merge(pop_data, on='Siruta')
    print(merged_data.head())

    columns_for_sum = merged_data.columns[1:-3]
    grouped_data = pd.DataFrame(merged_data.groupby('Judet')[columns_for_sum].sum())
    print(grouped_data.head())

    grouped_pop = pd.DataFrame(pop_data.groupby('Judet')['Populatie'].sum())
    print(grouped_pop)

    cer2 = grouped_data.merge(grouped_pop, on='Judet')
    cer2 = cer2.apply(func=lambda x: (x * 100000) / x.iloc[-1], axis=0)
    # for row in cer2.index:
    #     for col in cer2.columns[0:-1]:
    #         cer2[col] = cer2[col] * 100000 / cer2.loc[row, cer2.columns[-1]]
    print(cer2.head())


def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col].fillna(data[col].mean(), inplace=True)
    return data

def analiza_factoriala():
    caen_data = pd.read_csv('dataIN/CAEN2_2021_NSAL.csv')
    numeric_cleaned_data = clean_data(caen_data.select_dtypes(['float64', 'int64']))
    #print(numeric_cleaned_data)
    data_to_be_standartized = numeric_cleaned_data.iloc[:,1:]
    print(data_to_be_standartized)

    # Standartizare
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_to_be_standartized)
    #print(normalized_data)

    # Modelul Factorial
    nr_var = normalized_data.shape[1]
    # daca vrem rotatie rotation=  'varimax'
    model_Fact = FactorAnalyzer(n_factors=nr_var, rotation=None)
    F = model_Fact.fit(normalized_data)

    # Scoruri
    scoruri = model_Fact.transform(normalized_data)
   # print(scoruri)
    columns = ['F'+str(i+1) for i in range(0, normalized_data.shape[1])]
    df_scoruri = pd.DataFrame(scoruri, columns=columns)
    df_scoruri.to_csv('dataOUT/scoruri.csv')
    plt.figure(figsize=(10,7))
    plt.scatter(df_scoruri['F1'], df_scoruri['F2'])
    plt.title('Vizualizare scoruri')
    plt.show()

    # Barlett Test
    barlett_test = calculate_bartlett_sphericity(normalized_data)
    print('Barlett Test testul chow patrat: ', barlett_test[0])
    print('Barlett Test p-value: ', barlett_test[1])

    # KMO test
    kmo_test = calculate_kmo(normalized_data)
    print('KMO general: ', kmo_test[0])
    #print('KMO individual: ', kmo_test[1])


    # Varianta
    # [0] cata varianta explica fiecare factor
    variances = model_Fact.get_factor_variance()[0]
    print('Variances:' ,variances)

    # Corelatii
    corelatii = model_Fact.loadings_
    df_corelatii = pd.DataFrame(
        corelatii,
        columns=columns
    )
    df_corelatii.to_csv('dataOUT/corelatii.csv')

    # Diagrama de corelatie
    plt.figure(figsize=(10,7))
    sb.heatmap(corelatii)
   # plt.show()

    # Comunalitatii
    comunalitatii = model_Fact.get_communalities()
    df_comunalitatii = pd.DataFrame(
        data=comunalitatii,

        columns=['Comunalitati']
    )
    print(df_comunalitatii)
    sb.heatmap(df_comunalitatii, vmin=0, annot=True)
    plt.show()




analiza_factoriala()