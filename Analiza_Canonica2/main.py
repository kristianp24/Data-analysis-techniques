import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def cerinta1():
    industrie = pd.read_csv('dataIN/Industrie.csv')
    populatie = pd.read_csv('dataIN/PopulatieLocalitati.csv')
    merged_data = industrie.merge(populatie, on='Siruta')
    columns_to_be_dropped = ['Localitate_y', 'Judet']
    merged_data.drop(columns_to_be_dropped, inplace=True, axis=1)

    cer1 = merged_data.apply(func=lambda x: x[2:-1] / x[-1], axis=1)
    merged_data.iloc[:, 2:-1] = cer1
    merged_data.drop('Populatie', axis=1, inplace=True)
    merged_data.to_csv('dataOUT/cerinta1.csv')

def cerinta2():
    industrie = pd.read_csv('dataIN/Industrie.csv')
    populatie = pd.read_csv('dataIN/PopulatieLocalitati.csv')
    merged_data = industrie.merge(populatie, on='Siruta')
    columns_to_be_dropped = ['Localitate_y', 'Localitate_x', 'Populatie']
    merged_data.drop(columns_to_be_dropped, inplace=True, axis=1)
    columns_for_grouping = merged_data.columns[1:-1]
    grouped_data = merged_data.groupby(by='Judet')[columns_for_grouping].sum()
    max_Ids_Judet = grouped_data.apply(func=lambda x: x.idxmax(), axis = 1)
    cer2 = pd.DataFrame(max_Ids_Judet, columns=['Dominant Industry'])
    cer2.to_csv('dataOUT/cerinta2.csv')

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data
def canonical_analysys():
    data = pd.read_csv('dataIN/DataSet_34.csv')
    cleaned_data = clean_data(data.iloc[:, 1:])

    # Standartizare si grupare date
    scaler = StandardScaler()
    columns_setX = ['prodPorc','prodVite','prodOaieSiCapra','prodPasareDeCurte']
    columns_setY = ['consPorc','consVita','consumOaieSiCapra','consPasareDeCurte']
    X = pd.DataFrame(scaler.fit_transform(cleaned_data[columns_setX]), index=data['tari'], columns=columns_setX)
    Y = pd.DataFrame(scaler.fit_transform(cleaned_data[columns_setY]), index=data['tari'], columns=columns_setY)
    X.to_csv('dataOUT/Xstd.csv')
    Y.to_csv('dataOUT/Ystd.csv')

    X,Y = X.align(Y, join='inner', axis=0)

    # Modelul Canonic
    modelCanonic = CCA(n_components=2)

    # Scores
    X_c,Y_c = modelCanonic.fit_transform(X,Y)
    X_c_df = pd.DataFrame(X_c, columns=['C1','C2'], index=data['tari'])
    Y_c_df = pd.DataFrame(Y_c, columns=['C1','C2'], index=data['tari'])
    X_c_df.to_csv('dataOUT/z.csv')
    Y_c_df.to_csv('dataOUT/u.csv')

    # Corelatii factoriale
    corr_X = np.corrcoef(X.T, X_c.T)[:X.shape[1],X.shape[1]:]
    corr_Y = np.corrcoef(Y.T, Y_c.T)[:Y.shape[1], Y.shape[1]:]
    Rxu = pd.DataFrame(corr_Y, index=columns_setX, columns=['C1', 'C2'])
    Ryu = pd.DataFrame(corr_Y, index=columns_setX, columns=['C1','C2'])
    Rxu.to_csv('dataOUT/Rxu.csv')
    Ryu.to_csv('dataOut/Ryu.csv')

    # Biplot
    plt.figure(figsize=(8,7))
    plt.scatter(X_c[:, 0], X_c[:, 1], color = 'blue')
    plt.scatter(Y_c[:, 0], Y_c[:, 1], color = 'red')
    plt.title('Biplot')
    plt.show()



canonical_analysys()