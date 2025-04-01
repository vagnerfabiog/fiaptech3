# Projeto de Detecção de Anomalias em Dados SCADA

Este projeto realiza a detecção de comportamentos anômalos em sensores de pressão, utilizando dados históricos armazenados em um bucket S3 no formato Parquet. O pipeline inclui o pré-processamento, treinamento de um modelo de Machine Learning (XGBoost), e um dashboard interativo com Streamlit para visualização e predição em tempo real.

---

## 📁 Estrutura do Projeto

- `treinar_modelo_alerta.py` → script de treinamento do modelo com dados reais e sintéticos
- `dashboard.py` → aplicação com Streamlit para predição com o modelo treinado
- `modelo_alerta_xgboost_sintetico.pkl` → modelo salvo após o treinamento

---

## 🔍 Descrição do Treinamento (`treinar_modelo_alerta.py`)

1. **Coleta de dados** do S3 (`.parquet`) com autenticação segura.
2. **Limpeza** de valores nulos e negativos.
3. **Engenharia de features**:
   - `dif_media = media_hora - media_hora_30_dias`
   - `media_hora_relativa = media_hora / (media_hora_30_dias + 1)`
4. **Geração de dados sintéticos** de alerta para balancear a base.
5. **Treinamento com XGBoost**, utilizando `scale_pos_weight` para lidar com desbalanceamento.
6. **Avaliação com métricas** de classificação.
7. **Salvamento do modelo** com `joblib`.

---

## 📊 Aplicação(`app.py`)

- Leitura do modelo `.pkl`
- Interface para entrada de dados manuais ou via API
- Predição do alerta com base nas variáveis de entrada
- Exibição do resultado de forma clara e responsiva

---

## ▶️ Como Executar

### 1. Instalar dependências:

```bash
pip install -r requirements.txt
```

Ou diretamente:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn fastapi uvicorn joblib streamlit imbalanced-learn s3fs pyarrow xgboost
```

### 2. Treinar o modelo (opcional):

```bash
python treinar_modelo_alerta.py
```

### 3. Rodar o Aplicativo:

```bash
streamlit run app.py
```

---

## 📦 Requisitos

- Python 3.8+
- Acesso à AWS com credenciais válidas no Colab ou ambiente local
- Streamlit para rodar o dashboard

---

## 👨‍💻 Autor

Desenvolvido por Diego e Vagner como parte de um projeto acadêmico de Machine Learning com foco em detecção de anomalias em dados SCADA.
