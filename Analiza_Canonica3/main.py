import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def cerinta1():
    emissions = pd.read_csv('dataIN/emmissions.csv')
    c1 = emissions.apply(func=lambda x: x['AirEmiss':].sum(), axis=1)
    data = {'CountryCode': emissions['ThreeLetterCountryCode'],
            'Country': emissions['Country'],
            'Emisii_tone': c1}
    cer1 = pd.DataFrame(data)
    cer1.to_csv('dataOUT/cerinta1.csv')

def cerinta2():
    emissions = pd.read_csv('dataIN/emmissions.csv')
    pop = pd.read_csv('dataIN/PopulatieEuropa.csv')

    merged_data = emissions.merge(pop, on='ThreeLetterCountryCode')
    merged_data.drop(['Country_x', 'Country_y', 'Population'], inplace=True, axis=1)
    grouped_data = merged_data.groupby(by='Region')[emissions.columns[2:]].sum()
    c2 = grouped_data.apply(func=lambda x: x['AirEmiss':] / 100000, axis=1)
    c2.to_csv('dataOUT/cerinta2.csv', index=True)

def clean_data(data:pd.DataFrame):
    if data.isna().any().any():
        for col in data.columns:
            if data[col].isna().any():
                data[col].fillna(data[col].mean(), inplace=True)
    return data

def canonical_analysys():
    emissions = pd.read_csv('dataIN/emmissions.csv')
    # Setul de date nu e la fel ca cel dat in examen, asa ca aici doar impart datele cum vreau eu
    set_x = ['AirEmiss','Sulphur','Nitrogen','Ammonia']
    set_y = ['NonMeth','Partic','GreenGE_tone','GreenGIE_tone']

    # Clean and standartize
    scaler = StandardScaler()
    X = scaler.fit_transform(clean_data(emissions.loc[:, set_x]))
    Y = scaler.fit_transform(clean_data(emissions.loc[:, set_y]))

    # X,Y = X.align(Y, join='inner', axis=1)

    # Model Canonic
    modelCCA = CCA(n_components=2)
    z, u = modelCCA.fit_transform(X,Y)
    u_df = pd.DataFrame(u, columns=['C1', 'C2'])
    z_df = pd.DataFrame(z, columns=['C1', 'C2'])
    u_df.to_csv('dataOUT/u.csv')
    z_df.to_csv('dataOUT/z.csv')

    # Corelatii canonice
    corr_X = np.corrcoef(X.T, z.T)[:X.shape[1], X.shape[1]:]
    corr_Y = np.corrcoef(Y.T, u.T)[:Y.shape[1], Y.shape[1]:]
    corr_X_df = pd.DataFrame(corr_X, index=set_x, columns=['C1','C2'])
    corr_Y_df = pd.DataFrame(corr_Y, index=set_x, columns=['C1','C2'])
    r = pd.concat([corr_X_df, corr_Y_df], axis=0)
    r.to_csv('dataOUT/r.csv', index=True)


canonical_analysys()