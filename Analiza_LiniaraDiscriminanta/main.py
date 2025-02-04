import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split

def cerinta1():
    buget = pd.read_csv('dateIN/Buget.csv')
    venituri = buget.apply(func=lambda x: x[2:6].sum(), axis = 1)
    cheltuieli = buget.apply(func=lambda x: x[7:].sum(), axis=1)
    data = {
        'Siruta': buget['Siruta'],
        'Localitate': buget['Localitate'],
        'Venituri': venituri,
        'Cheltuieli': cheltuieli
    }
    pd.DataFrame(data).to_csv('dateOUT/cerinta1.csv')

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data
def discriminant_analysys():
    pacienti = pd.read_csv('dateIN/Pacienti.csv')
    pacienti_apply = pd.read_csv('dateIN/Pacienti_apply.csv')

    pacienti_cleaned = clean_data(pacienti.iloc[:,:-1])
    pacienti_apply_cleaned = clean_data(pacienti_apply)

    # Var INdependente
    X = pacienti_cleaned
    # Var Dependenta
    Y = pacienti['DECISION']

    # Set de test + Set de antrenament
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.35, random_state=42)

    # Model LDA
    modelLDA = LinearDiscriminantAnalysis()
    modelLDA.fit(X_train, Y_train)

    # Predictia
    predict_Y = modelLDA.predict(X_test)
    predict_Y_df = pd.DataFrame(data={
        'Real': Y_test,
        'Predicted': predict_Y
    })
    predict_Y_df.to_csv('dateOUT/predict.csv')

    # Matricea de confuzie
    matrice_confuzie = confusion_matrix(Y_test, predict_Y)
    print('Matrice confuzie: ', matrice_confuzie)

    # Acuratatea globala
    gA = accuracy_score(Y_test, predict_Y)
    print('Acuratatea globala: ', gA)

    # Acuratatea Medie
    a_per_class = matrice_confuzie.diagonal() / matrice_confuzie.sum(axis=1)
    a_Medie = np.mean(a_per_class)
    print('Acuratatea medie: ', a_Medie)



discriminant_analysys()