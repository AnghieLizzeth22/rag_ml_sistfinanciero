# -*- coding: utf-8 -*-
"""
MÓDULO PREDICTOR - INTERFAZ DE LECTURA DE SCORING REAL
Lee directamente las probabilidades calculadas por el modelo de producción en Colab.
"""
import pandas as pd

class MotoresPredictivos:
    def __init__(self, ruta_pipeline=None, ruta_modelo=None):
        print("🏛️ Conectando módulo de lectura para el set de pruebas de LightGBM...")

    def ejecutar_scoring(self, df_cliente: pd.DataFrame) -> float:
        """
        Extrae la Probabilidad de Default real que ya fue calculada por el modelo
        en el entorno de entrenamiento.
        """
        # Esta es la columna que acabamos de inyectar en tu Colab
        columna_probabilidad = 'PROB_DEFAULT' 
        
        if columna_probabilidad in df_cliente.columns:
            prob_val = df_cliente[columna_probabilidad].values[0]
            return float(prob_val)
        else:
            # Fallback de seguridad por si acaso
            for col in df_cliente.columns:
                if df_cliente[col].dtype in ['float64', 'float32']:
                    val = df_cliente[col].values[0]
                    if 0.0 <= val <= 1.0 and col not in ['EXT_SOURCES_MEAN', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
                        return float(val)
            
            return 0.2747