import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

def cerinta1():
    vot = pd.read_csv('dataIN/Vot.csv')
    cer1 = vot.apply(func=lambda x: x[3:] * 100 / x['Votanti_LP'], axis=1)
    cer1_df = vot.iloc[:,0:1].merge(cer1, left_index=True, right_index=True)
    cer1.to_csv('dataOUT/cerinta1.csv')

def cerinta2():
    vot = pd.read_csv('dataIN/Vot.csv')
    coduri = pd.read_csv('dataIN/Coduri_localitati.csv')

    merged_data = vot.merge(coduri, on='Siruta')
    merged_data.drop(['Siruta', 'Localitate_x', 'Mediu', 'Localitate_y'], inplace=True, axis=1)
    grouped_data = pd.DataFrame(merged_data.groupby('Judet')[vot.columns[2:]].sum())
    cer2 = grouped_data.apply(func=lambda x: x[1:] * 100 / x['Votanti_LP'], axis=1)
    print(cer2.head())

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data

def canonical_analysys():
    vot = pd.read_csv('dataIN/Vot.csv')
    cleaned_data = clean_data(vot.iloc[:,3:])
    # Impartirea setului de date
    barbati = ['Barbati_25-34','Barbati_35-44','Barbati_45-64','Barbati_65_']
    femei = ['Femei_18-24','Femei_35-44','Femei_45-64','Femei_65_']

    barbati_df = cleaned_data.loc[:,barbati]
    femei_df = cleaned_data.loc[:,femei]

    # Cele 2 seturi trebuie sa fie egale
    barbati_df, femei_df = barbati_df.align(femei_df, join='inner', axis=0)

    # Construim modelul Canonic
    model_C = CCA(n_components=2)
    # Scorurile canocnice
    barbati_c, femei_C = model_C.fit_transform(barbati_df, femei_df)
    barbati_c_df = pd.DataFrame(barbati_c, columns=['C1','C2'])
    femei_c_df = pd.DataFrame(femei_C, columns=['C1','C2'])
    barbati_c_df.to_csv('dataOUT/z.csv')
    femei_c_df.to_csv('dataOUT/y.csv')

    # Corelatii canonice
    cor_canonice = model_C.score(barbati_df, femei_df)
    print('Corelatii canoncie: ', cor_canonice)

    # Coef de corelatie
    corr_x = np.corrcoef(barbati_df.T, barbati_c.T)[:barbati_df.shape[1], barbati_df.shape[1]:]
    corr_y = np.corrcoef(femei_df.T, femei_C.T)[:femei_df.shape[1], femei_df.shape[1]:]
    corr_x_df = pd.DataFrame(corr_x, index=barbati, columns=['C1','C2'])
    corr_y_df = pd.DataFrame(corr_y, index=femei, columns=['C1','C2'])
    r = pd.concat([corr_x_df, corr_y_df], axis=0)
    r.to_csv('dataOUT/r.csv')

    # Trasare plot primele radacine canonice
    plt.figure(figsize=(10,7))
    plt.scatter(barbati_c[:,0], barbati_c[:, 1], color='red', label='Prima radacina canonica')
    plt.scatter(femei_C[:, 0], femei_C[:, 1], color='red', label='A doua radacina canonica')
    plt.title('Plot primele 2 radacini canonice')
    plt.xlabel('Componenta barbati')
    plt.ylabel('Componenta femei')
    plt.show()







canonical_analysys()