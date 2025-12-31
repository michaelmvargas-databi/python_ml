import streamlit as stl
import joblib
import plotly.express as px
import pandas as pd
from io import BytesIO

modelo = joblib.load("modelo_riesgo_opt.joblib")
stl.set_page_config("Modelo de riesgo","🤖")

stl.title("Evaluador de credito")

# Ingreso de datos del usuario

loan_percent_income = stl.number_input("Ingrese el ratio ingreso/prestamo",value=0.17)
loan_int_rate = stl.number_input("Ingrese la tasa de interes",value = 11)
person_income = stl.number_input("Ingrese el nivel de ingresos($)",value = 66000)

if stl.button("Calcular Riesgo"):
    probs = modelo.predict_proba([[loan_percent_income,loan_int_rate,person_income]])
    riesgo = probs[0][1]
    stl.write(f"Riesgo Calculado {riesgo:.1%}")
    grafica_pie = px.pie(names=["riesgo","confianza"],values=[riesgo,1-riesgo],
                         color=["riesgo","confianza"],color_discrete_map={'riesgo':'red','confianza':'blue'})
    stl.plotly_chart(grafica_pie)

archivo = stl.file_uploader("Evaluación masiva",type=['csv','xlsx'])
if archivo is not None:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo)
    elif archivo.name.endswith(".xlsx"):
        df = pd.read_excel(archivo)

    stl.success(f"Archivo cargado correctamente\nFilas: {len(df)}")
    #stl.dataframe(df.head())
    df['riesgo'] = modelo.predict_proba(df)[:,1]
    
    #aqui creamos el archivo en la ram
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer,index=False,sheet_name='Reporte')
    
    #aqui seleccionamos el documento
    output.seek(0)

    stl.download_button(label='Descargar EXCEL',data=output,file_name="reporte_masivo.xlsx")
    
    stl.dataframe(df.head())
    grafica_histograma = px.histogram(x=df['riesgo'])
    stl.plotly_chart(grafica_histograma)

