"""
Script principal do sistema de trading automatizado.
Este script coordena todas as operações do sistema, desde a coleta de dados
até a execução do backtest e análise de resultados.
"""

# 🔧 Bibliotecas básicas
import pandas as pd
import numpy as np
import ta  # Biblioteca para indicadores técnicos
import time

# 📊 Machine Learning e processamento
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE  # Para balanceamento de classes

# 🧠 Seu robô
from classeRobo import RoboTrading

# 📈 Dados
from processar_dados import coletar_dados_15min, adicionar_indicadores

# 📋 Filtros
from aplicar_filtros import aplicar_filtros_tecnicos

# 🔁 Backtest
from backtest_utils import backtest_avancado
from backtest_agressivo import backtest_agressivo

# Analise de desempenho
from analisar_desempenho import calculo_desempenho
from visualizar_trades import plot_trades, plot_retornos, plot_equity_curve, plot_drawdown

# Configurações e Logging
from config import (
    TRADING_CONFIG,
    MODEL_CONFIG,
    BACKTEST_CONFIG,
    VISUALIZATION_CONFIG,
    YFINANCE_CONFIG
)
from logger import logger, log_trade, log_backtest_results, log_error

def main():
    """
    Função principal que executa o fluxo completo do sistema.
    """
    try:
        # 1. Coleta e preparação dos dados
        logger.info("Iniciando coleta de dados históricos...")
        df = coletar_dados_15min(YFINANCE_CONFIG['ticker'])
        logger.info("Adicionando indicadores técnicos...")
        df = adicionar_indicadores(df, YFINANCE_CONFIG['ticker'])

        # 2. Inicialização e treinamento do modelo
        logger.info("Inicializando robô de trading...")
        robo = RoboTrading(
            capital_inicial=TRADING_CONFIG['capital_inicial'],
            risco_por_trade=TRADING_CONFIG['risco_por_trade'],
            stop_loss=TRADING_CONFIG['stop_loss_pct'],
            take_profit=TRADING_CONFIG['take_profit_pct'],
            trailing_stop=TRADING_CONFIG['trailing_stop']
        )
        
        logger.info("Preparando dados para treinamento...")
        df_preparado = robo.preparar_dados(df)
        
        logger.info("Treinando modelo...")
        probs = robo.treinar_modelo(df_preparado)

        # 3. Análise das previsões
        logger.info(f"Total de previsões com alta confiança (> {MODEL_CONFIG['confidence_threshold']}): {(probs > MODEL_CONFIG['confidence_threshold']).sum()}")
        logger.info("Amostra de probabilidades:")
        logger.info(df_preparado[['close']].iloc[-10:].assign(proba=probs[-10:]))

        # 4. Execução do backtest
        logger.info("Executando backtest...")
        metricas, df_test, entradas, saidas = robo.executar_backtest(
            df_preparado, 
            probs, 
            modo=BACKTEST_CONFIG['modo_padrao']
        )

        # 5. Visualização dos resultados
        logger.info("Plotando trades...")
        fig = plot_trades(
            df_test, 
            entradas, 
            saidas,
            title='Visualização de Trades'
        )
        fig.show()

        # 6. Exibição das métricas
        log_backtest_results(metricas)

    except Exception as e:
        log_error("Erro durante a execução do sistema", e)
        raise

if __name__ == "__main__":
    main()