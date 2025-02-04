import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt

def cerinta1():
    data = pd.read_csv('dataIN/E_NSAL_2008-2021.csv', index_col='SIRUTA')
    cer2 = data.apply(lambda x: x.idxmax(), axis=1)
    pd.DataFrame(cer2, columns=['Anul']).to_csv('dataOUT/cerinta1.csv')
    print(cer2.head())

def cerinta2():
    data = pd.read_csv('dataIN/E_NSAL_2008-2021.csv')
    pop = pd.read_csv('dataIN/PopulatieLocalitati.csv')
    merged_df = data.merge(pop, on='Siruta').drop(['Localitate'], axis=1)
    columns = list(data.columns[1:]) + ['Populatie']
    grouped_df= merged_df.groupby(by='Judet')[columns].sum()
    c = grouped_df.apply(func=lambda x: np.round(x[0:-1] / x[-1],3), axis=1)
    rate_medii = pd.Series(c.apply(func=lambda x: np.round(x[0:].mean(), 3), axis=1))
    c['RataMedie'] = rate_medii
    c.sort_values(by='RataMedie', ascending=False, inplace=True)
    c.to_csv('dataOUT/cerinta2.csv')

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data

def discriminant_analysys():
    pacienti = pd.read_csv('dataIN/Pacienti.csv')
    pacienti_apply = pd.read_csv('dataIN/Pacienti_apply.csv')

    pacienti_cleaned = clean_data(pacienti.iloc[:, 1:-1])
    pacienti_apply_cleaned = clean_data(pacienti_apply.iloc[:, 1:])


    X_train = pacienti_cleaned
    Y_train = pacienti['DECISION']

    X_test = pacienti_apply_cleaned
    Y_test =[]
    # # Model te testare + antrenare
    # X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.35, random_state=42)


    #Model LDA
    modeLDA = LinearDiscriminantAnalysis()

    # Scoruri pentru antrenare
    scoruri = modeLDA.fit_transform(X_train, Y_train)
    print('Scoruri antrenare: ', scoruri)

    # Grafic
    plt.scatter(scoruri[:,0], scoruri[:,1], color='red')
    plt.show()

    # Scoruri pentru test
    scoruri_test = modeLDA.transform(X_test)
    print('Scoruri test:', scoruri_test)
    print(scoruri.shape)



discriminant_analysys()
