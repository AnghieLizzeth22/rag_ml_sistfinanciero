# -*- coding: utf-8 -*-
"""
APLICACIÓN PRINCIPAL - CORE RISK SYSTEM (ML + RAG)
Versión optimizada con Ingresos Totales y Extracción Segura
"""
import streamlit as st
import pandas as pd
import os
from src.predictor import MotoresPredictivos
from src.engine_rag import GeneradorExplicacionesRAG

# 1. INYECTAR LA LLAVE DIRECTAMENTE AQUÍ
os.environ["GOOGLE_API_KEY"] = "AIzaSyAkN3UGXRrFfMyWpR88fiv0dxdhNHekspM"

st.set_page_config(page_title="Core Risk System | ML + RAG", layout="wide")

st.title("🏛️ Sistema de Admisión de Riesgo Crediticio Híbrido")
st.subheader("Maestría en Ciencia de Datos / Ingeniería de Sistemas - UNI")
st.markdown("---")

@st.cache_resource
def inicializar_motor_ml():
    return MotoresPredictivos()

try:
    predictor = inicializar_motor_ml()
    st.sidebar.success("✅ Modelo LightGBM cargado localmente.")
    st.sidebar.success("🔑 Credenciales de Gemini configuradas.")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar LightGBM: {str(e)}")
    st.stop()

ruta_datos = 'data/X_test_sample.csv'

if not os.path.exists(ruta_datos):
    st.error(f"❌ Archivo central '{ruta_datos}' no localizado.")
else:
    df_db = pd.read_csv(ruta_datos)
    
    st.markdown("### 🔍 Módulo de Consulta de Solicitudes")
    id_buscado = st.number_input("Ingrese el Identificador Único del Cliente (SK_ID_CURR):", step=1, value=int(df_db['SK_ID_CURR'].iloc[0]))
    
    if st.button("Evaluar Solicitud de Crédito"):
        # 1. Filtramos la fila exacta del deudor consultado
        registro_cliente = df_db[df_db['SK_ID_CURR'] == id_buscado].copy()
        
        if registro_cliente.empty:
            st.error(f"❌ El identificador {id_buscado} no se encuentra registrado.")
        else:
            # 2. Ejecución dinámica del modelo predictivo sobre el cliente seleccionado
            probabilidad_mora = predictor.ejecutar_scoring(registro_cliente)
            
            # --- EXTRACCIÓN DEFENSIVA DE VARIABLES (Evita KeyError y maneja NaNs) ---
            
            # Reemplazamos la variable eliminada por Ingresos Totales (AMT_INCOME_TOTAL)
            ingresos = float(registro_cliente['AMT_INCOME_TOTAL'].values[0]) if 'AMT_INCOME_TOTAL' in registro_cliente.columns and pd.notna(registro_cliente['AMT_INCOME_TOTAL'].values[0]) else 0.0
            
            # Días de retraso (MAX_RETRASO_6M o alternativas como SK_DPD)
            retraso_col = 'MAX_RETRASO_6M' if 'MAX_RETRASO_6M' in registro_cliente.columns else ('SK_DPD' if 'SK_DPD' in registro_cliente.columns else None)
            retraso = float(registro_cliente[retraso_col].values[0]) if retraso_col and pd.notna(registro_cliente[retraso_col].values[0]) else 0.0
            
            # Edad y Score Externo Promedio
            edad = float(registro_cliente['EDAD_AÑOS'].values[0]) if 'EDAD_AÑOS' in registro_cliente.columns and pd.notna(registro_cliente['EDAD_AÑOS'].values[0]) else 35.0
            ext_mean = float(registro_cliente['EXT_SOURCES_MEAN'].values[0]) if 'EXT_SOURCES_MEAN' in registro_cliente.columns and pd.notna(registro_cliente['EXT_SOURCES_MEAN'].values[0]) else 0.5
            
            # --- DESPLIEGUE DE MÉTRICAS EN PANTALLA ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="Probabilidad de Default (PD)", value=f"{probabilidad_mora * 100:.2f}%")
            col2.metric(label="Ingresos Totales (Income)", value=f"${ingresos:,.2f}")
            col3.metric(label="Mora Reciente (6M)", value=f"{retraso:.0f} días")
            col4.metric(label="Score Externo Promedio", value=f"{ext_mean:.3f}")
            
            st.markdown("---")
            st.markdown("### 📝 Dictamen Automatizado de Auditoría de Riesgos (RAG)")
            
            # 3. Activación del pipeline semántico con Gemini 2.5 Flash
            with st.spinner("✍️ Analizando base de conocimiento y redactando dictamen..."):
                try:
                    generador_rag = GeneradorExplicacionesRAG()
                    
                    # Estructura de datos dinámica que se envía al prompt del RAG
                    payload_explicacion = f"""
                    Resultados del scoring cuantitativo:
                    - Probabilidad de Default calculada por el modelo: {probabilidad_mora * 100:.2f}%
                    - Identificador del deudor: {id_buscado}
                    - Ingresos totales demostrados: ${ingresos:,.2f}
                    - Días de mora registrados en los últimos 6 meses: {retraso} días
                    - Edad cronológica del solicitante: {edad:.1f} años
                    - Score crediticio consolidado (Burós Externos): {ext_mean:.4f}
                    """
                    
                    dictamen_final = generador_rag.generar_reporte(payload_explicacion)
                    st.info(dictamen_final)
                    
                except Exception as error_rag:
                    st.error(f"❌ Error en el motor RAG: {str(error_rag)}")