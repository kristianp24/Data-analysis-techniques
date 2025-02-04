import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col].fillna(data[col].mean(), inplace=True)
    return data


def canonical_analysys():
    religious = pd.read_csv('dataIN/ReligiousP.csv')
    structura_pop = pd.read_csv('dataIN/StructuraPopulateiPeGrupe_Localitati.csv')
    print (len(religious.columns[3:]), len(structura_pop.columns[3:]))

    # O sa folosesc date doar din religious
    set1_columns = list(religious.columns[3:13])
    set2_columns = list(religious.columns[13:-1])

    X = clean_data(religious[set1_columns])
    Y = clean_data(religious[set2_columns])

    # Construim modelul Canonic
    modelCCA = CCA(n_components=2)

    # Scoruri
    scoruri_X, sroruri_Y= modelCCA.fit_transform(X,Y)


    # Correlatii
    corr_X = np.corrcoef(X.T, scoruri_X.T)[:len(set1_columns), len(set1_columns):]
    corr_Y = np.corrcoef(Y.T, sroruri_Y.T)[:len(set2_columns), len(set2_columns):]
    print("Corr X: ")
    print(corr_X)

    #Biplot
    plt.scatter(scoruri_X[:,0], scoruri_X[:, 1], color = 'r')
    plt.scatter(sroruri_Y[:, 0], sroruri_Y[:, 1], color = 'blue')
    plt.show()



canonical_analysys()