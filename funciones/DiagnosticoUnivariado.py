##############################################################################################
# Analisis_Univariado 
##############################################################################################

import pandas as pd
import numpy as np

def variables_categoricas(df):
    return df.select_dtypes(include=['object', 'category'])

def variables_numericas(df):
    return df.select_dtypes(include=[np.number])

def DiagnosticoUnivariado(df):
    """
    Realiza un escaneo 360 de la salud de las variables del DataFrame.
    """
    resultados = []
    for col in df.columns:
        tipo = df[col].dtype
        nulos = df[col].isnull().sum()
        pct_nulos = (nulos / len(df)) * 100
        n_unicos = df[col].nunique()
        
        num_outliers = 0
        pct_outliers = 0
        concentracion = 0
        
        if np.issubdtype(tipo, np.number):
            data = df[col].dropna()
            if not data.empty:
                Q1, Q3 = np.percentile(data, [25, 75])
                IQR = Q3 - Q1
                lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                num_outliers = ((data < lim_inf) | (data > lim_sup)).sum()
                pct_outliers = (num_outliers / len(data)) * 100
                concentracion = data.value_counts(normalize=True).iloc[0] * 100
        else:
            if not df[col].dropna().empty:
                concentracion = df[col].value_counts(normalize=True).iloc[0] * 100

        resultados.append([col, tipo, n_unicos, nulos, pct_nulos, num_outliers, pct_outliers, concentracion])
    
    resumen = pd.DataFrame(resultados, columns=[
        'Variable', 'Tipo', 'Unicos', 'Nulos', '%_Nulos', 'Num_Outliers', '%_Outliers', '%_Concentracion'
    ])
    return resumen.set_index('Variable')

# @copyright Anghie Chilon  