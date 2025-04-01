import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ✅ Carregar dados do S3
from google.colab import userdata

df = pd.read_parquet(
    "s3://techf3/scada/dados_pressao.parquet",
    storage_options={
        "key": userdata.get('keyaws'),
        "secret": userdata.get('secretAws'),
        "client_kwargs": {"region_name": "us-east-1"}
    }
)

# ✅ Limpeza e tratamento
df = df.dropna()
df = df[df['media_hora'] >= 0]

df['dif_media'] = df['media_hora'] - df['media_hora_30_dias']
df['media_hora_relativa'] = df['media_hora'] / (df['media_hora_30_dias'].clip(lower=0.1) + 1)
df['alerta'] = df['alerta'].astype('category').cat.codes

# ✅ Gerar dados sintéticos de alerta
novos_alertas = pd.DataFrame({
    'latitude': [-7.12]*30,
    'longitude': [-34.88]*30,
    'hora': np.random.randint(0, 24, 30),
    'media_hora': np.random.uniform(30, 50, 30),
    'media_hora_30_dias': np.random.uniform(5, 15, 30),
    'desvio_padrao_30_dias': np.random.uniform(5, 10, 30)
})
novos_alertas['dif_media'] = novos_alertas['media_hora'] - novos_alertas['media_hora_30_dias']
novos_alertas['media_hora_relativa'] = novos_alertas['media_hora'] / (novos_alertas['media_hora_30_dias'] + 1)
novos_alertas['alerta'] = 1

# ✅ Combinar dados reais e sintéticos
df_aug = pd.concat([df, novos_alertas], ignore_index=True)

# ✅ Seleção de variáveis
X = df_aug[['hora', 'media_hora', 'media_hora_30_dias',
            'desvio_padrao_30_dias', 'dif_media', 'media_hora_relativa']]
y = df_aug['alerta']

# ✅ Divisão treino/teste com estratificação
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# ✅ Treinamento do modelo com XGBoost
model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss')
model.fit(X_train, y_train)

# ✅ Avaliação do modelo
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, model.predict(X_test)))
print("✅ Acurácia:", accuracy_score(y_test, model.predict(X_test)))

# ✅ Salvar modelo
joblib.dump(model, "/modelo_alerta_xgboost_sintetico.pkl")
print("\n📁 Modelo salvo como: modelo_alerta_xgboost_sintetico.pkl")
