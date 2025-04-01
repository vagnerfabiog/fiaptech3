import streamlit as st
import pandas as pd
import joblib

# Carregar modelo treinado
model = joblib.load("modelo_alerta_xgboost_sintetico.pkl")

# Interface
st.title("Previsão de Alertas - SCADA")

# Campos de entrada
latitude = st.number_input("Latitude", value=-7.12)
longitude = st.number_input("Longitude", value=-34.87)
hora = st.slider("Hora", 0, 23, 12)
media_hora = st.number_input("Média da Última Hora", value=10.0)
media_hora_30_dias = st.number_input("Média dos Últimos 30 Dias", value=15.0)
desvio_padrao_30_dias = st.number_input("Desvio Padrão", value=3.0)

# Botão
if st.button("🔍 Prever Alerta"):
    data = pd.DataFrame([[latitude, longitude, hora, media_hora, media_hora_30_dias, desvio_padrao_30_dias]],
                        columns=['latitude', 'longitude', 'hora', 'media_hora', 'media_hora_30_dias', 'desvio_padrao_30_dias'])

    # Novas features que o modelo precisa
    data['dif_media'] = data['media_hora'] - data['media_hora_30_dias']
    data['media_hora_relativa'] = data['media_hora'] / (data['media_hora_30_dias'] + 1)

    # Obter probabilidade do alerta
    proba = model.predict_proba(data)[0][1]

    # Testar limiar mais baixo
    threshold = 0.15  # ajuste aqui se quiser testar 0.1 ou 0.2 também
    prediction = int(proba >= threshold)

    st.markdown(f"### Resultado: {'🟥 ALERTA' if prediction == 1 else '✅ Normal'}")
    st.markdown(f"**Probabilidade de alerta:** {proba:.4f}")
    st.markdown(f"**Threshold usado:** {threshold}")

    if prediction == 1:
        st.warning("⚠️ Atenção: comportamento anômalo identificado!")
    else:
        st.info("Sem alerta previsto. Continue monitorando.")
