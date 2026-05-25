# -*- coding: utf-8 -*-
import os
import docx2txt
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class GeneradorExplicacionesRAG:
    def __init__(self, ruta_txt='docs/manual_politicas.txt', ruta_docx='docs/20140926_res_3780-2011.docx'):
        texto_corporativo = ""
        
        if os.path.exists(ruta_txt):
            with open(ruta_txt, 'r', encoding='utf-8') as f:
                texto_corporativo += f.read() + "\n\n"
        
        if os.path.exists(ruta_docx):
            texto_sbs = docx2txt.process(ruta_docx)
            texto_corporativo += texto_sbs
            
        if not texto_corporativo.strip():
            raise ValueError("❌ Error: La base de conocimiento está vacía.")
            
        chunks = [c.strip() for c in texto_corporativo.split("\n\n") if len(c.strip()) > 20]
        
        # 1. Fase de Vectores: Usamos HuggingFace localmente (esto solucionó el primer error)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_texts(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        self.template = """
        Eres un Oficial de Riesgos Senior y Asesor Financiero en una institución regulada por la SBS. 
        Tu labor es emitir un dictamen ejecutivo en español que explique de forma sencilla y pedagógica la decisión del modelo predictivo al cliente y le brinde recomendaciones viables.

        CONTEXTO REGULATORIO, NORMAS DE LA SBS Y DICCIONARIO DE VARIABLES:
        {context}

        MÉTRICAS REALES OBTENIDAS DEL PERFIL DEL CLIENTE EVALUADO:
        {datos_cliente}

        INSTRUCCIONES CRÍTICAS DE ANÁLISIS:
        1. Evalúa dinámicamente las métricas del cliente recibidas en 'datos_cliente' (Riesgo/PD, Mora Reciente, Uso de Línea, etc.). NO te bases en un porcentaje fijo.
        2. Determina si el cliente presenta comportamiento moroso activo o potencial analizando los días de atraso en el último semestre y su score externo.
        3. Traduce las siglas técnicas a términos de negocio usando el diccionario proveído (ej. no uses 'MAX_RETRASO_6M' en los títulos, llámalo 'Días de atraso en los últimos 6 meses').
        4. Sé directo: Dile al usuario por qué el modelo tomó esa decisión y qué significa su porcentaje de riesgo en el mundo real.

        ESTRUCTURA OBLIGATORIA DEL DICTAMEN (Usa formato Markdown):
        
        ## 🏛️ INFORME DE EVALUACIÓN DE CRÉDITO
        ---
        ### 1. ESTADO DE LA SOLICITUD
        * **RESULTADO:** [Determina dinámicamente si es APROBADO, RECHAZADO u OBSERVADO según sus métricas de riesgo y morosidad].
        * **Índice de Riesgo Estimado (Probabilidad de Default):** [Coloca aquí el porcentaje de PD real del cliente]%
        
        ### 2. ¿POR QUÉ SE TOMÓ ESTA DECISIÓN? (Análisis de Morosidad y Riesgo)
        * **Evaluación de Aatrasos Financieros:** [Explica de forma clara cuántos días de mora real tiene el cliente en el último semestre y cómo afecta eso a su evaluación].
        * **Evaluación de Respaldo Externo:** [Analiza el Score Externo Promedio del cliente según los burós de crédito y tradúcelo a si tiene buena o mala reputación en el sistema].
        * **Conclusión del Modelo:** [Explica de forma sencilla qué significa su porcentaje de riesgo actual para el banco].

        ### 3. PLAN DE ACCIÓN Y RECOMENDACIONES FINANCIERAS (¿Qué debe hacer el cliente?)
        * **Para corregir su situación actual:** [Brinda un consejo financiero personalizado basado en sus debilidades encontradas (ej. si tiene días de mora, aconseja regularizar fechas de pago; si su score es bajo, aconseja no sobregirarse)].
        * **Para mejorar su perfil a mediano plazo:** [Brinda una estrategia clara para que el cliente pueda mejorar su calificación ante las centrales de riesgo en los próximos meses].

        ---
        *Unidad de Posgrado FIEECS - Universidad Nacional de Ingeniería*
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        
        # Usamos la versión fundacional 1.0 que está habilitada para el 100% de las cuentas
        # Conectamos con la versión exacta que Google habilitó en tu cuenta
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
        self.chain = (
            {"context": self.retriever, "datos_cliente": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generar_reporte(self, payload_texto: str) -> str:
        return self.chain.invoke(payload_texto)