# 🔧 Bibliotecas básicas
import pandas as pd
import numpy as np
import ta
import time

# 📊 Machine Learning e processamento
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# 🧠 Seu robô
from classeRobo import RoboTrading

# 📈 Dados
from processar_dados import coletar_dados_15min, adicionar_indicadores

# 📋 Filtros
from aplicar_filtros import aplicar_filtros_mercado

# 🔁 Backtest
from backtest_utils import backtest_avancado
from backtest_agressivo import backtest_agressivo

# Analise de desempenho
from analisar_desempenho import calculo_desempenho
from visualizar_trades import plotar_trades_completos

ticker = "BTC-USD"
df = coletar_dados_15min(ticker)
df = adicionar_indicadores(df, ticker)

robo = RoboTrading()
df_preparado = robo.preparar_dados(df)
probs = robo.treinar_modelo(df_preparado)
print(f"Total de previsões com alta confiança (> 0.6): {(probs > 0.6).sum()}")
print("Amostra de probabilidades:")
print(df_preparado[['close']].iloc[-10:].assign(proba=probs[-10:]))
metricas, df_test, entradas, saidas = robo.executar_backtest(df_preparado, probs, modo='super_agressivo')

plotar_trades_completos(df_test, entradas, saidas)

print("Métricas do backtest:")
for k, v in metricas.items():
    print(f'{k}: {v:.2f}')