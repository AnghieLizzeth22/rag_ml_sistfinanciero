##############################################################################################
# Analisis_Univariado 
##############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def grap_variable_vs_target(df, variable, tipo='num'):
    """
    Grafica la distribución de una variable separada por el TARGET.
    tipo='num' para numéricas (KDE plot)
    tipo='cat' para categóricas (Gráfico de barras apiladas)
    """
    plt.figure(figsize=(8, 5))
    
    if tipo == 'num':
        # Para variables continuas 
        sns.kdeplot(data=df, x=variable, hue='TARGET', fill=True, common_norm=False, palette='viridis')
        plt.title(f'Distribución de {variable} según Riesgo (TARGET)')
        
    elif tipo == 'cat':
        # Para variables de texto o binarias 
        # Calculamos el % de morosidad por categoría
        prop_mora = df.groupby(variable)['TARGET'].mean().reset_index()
        prop_mora['TARGET'] = prop_mora['TARGET'] * 100
        
        sns.barplot(data=prop_mora, x=variable, y='TARGET', palette='Reds')
        plt.title(f'% de Morosidad por {variable}')
        plt.ylabel('% de Morosidad (TARGET=1)')
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.show()

# @copyright Anghie Chilon  
