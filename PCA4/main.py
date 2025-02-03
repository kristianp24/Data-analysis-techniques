import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sb
import matplotlib.pyplot as plt

def cerinta1():
    rata = pd.read_csv('dataIN/Rata.csv')
    media_RS = rata['RS'].mean()

    cer1 = rata[rata['RS'] < media_RS][['Three_Letter_Country_Code', 'Country_Name', 'RS']].sort_values(by='RS', ascending=False)
    cer1.to_csv('dataOUT/cerinta1.csv')



def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
    return data
def pca_analysys():
    data = pd.read_csv('dataIN/Rata.csv')
    cleaned_data = clean_data(data.iloc[:, 2:])

    # Standartizare
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)

    # Model pca
    model_pca = PCA()
    # scoruri
    scoruri = model_pca.fit_transform(normalized_data)

    varianta = model_pca.explained_variance_
    varianta_cum = np.cumsum(varianta)
    varianta_ratio = model_pca.explained_variance_ratio_
    varianta_cum_ratio = np.cumsum(varianta_ratio)

    B1 = {'Varianta_componentelor': varianta,
          'Vaianta_cumulata': varianta_cum,
          'Procentul_variante:': varianta_ratio,
          'Procentul cumulat': varianta_cum_ratio}
    B1_df = pd.DataFrame(B1)
    B1_df.to_csv('dataOUT/Varianta.csv')

   # Corelatii
    corr = np.corrcoef(normalized_data.T, scoruri.T)[:len(cleaned_data.columns), len(cleaned_data.columns):]
    corr_df = pd.DataFrame(corr, index=cleaned_data.columns, columns=['PC'+str(i+1) for i in range(len(cleaned_data.columns))])
    print(corr_df)
    # B3
    sb.heatmap(corr_df)
    plt.show()

    # B2
    plt.figure(figsize=(8,5))
    plt.plot(varianta)
    plt.axhline(1, color='red')
    plt.show()



pca_analysys()