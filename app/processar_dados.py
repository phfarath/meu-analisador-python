"""
Módulo para processamento e preparação de dados.
Implementa funções para coleta, limpeza e transformação de dados de mercado.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
from logger import logger
import os

def coletar_dados_15min(ticker, dias=30):
    """
    Coleta dados históricos do Yahoo Finance em intervalos de 15 minutos.
    
    Args:
        ticker (str): Símbolo do ativo (ex: 'AAPL')
        dias (int): Número de dias de dados históricos
        
    Returns:
        pd.DataFrame: DataFrame com dados OHLCV
    """
    logger.info(f"Coletando dados de {ticker} para os últimos {dias} dias")
    
    try:
        # Definir período
        fim = datetime.now()
        inicio = fim - timedelta(days=dias)
        
        # Coletar dados
        dados = yf.download(
            ticker,
            start=inicio,
            end=fim,
            interval='15m'
        )
        
        # Renomear colunas
        dados.columns = ['open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"Dados coletados com sucesso: {len(dados)} registros")
        return dados
        
    except Exception as e:
        logger.error(f"Erro ao coletar dados: {str(e)}")
        raise

def adicionar_indicadores(df, ticker):
    """
    Adiciona indicadores técnicos ao DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame com dados OHLCV
        ticker (str): Símbolo do ativo
        
    Returns:
        pd.DataFrame: DataFrame com indicadores adicionados
    """
    logger.info(f"Adicionando indicadores técnicos para {ticker}")
    
    try:
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Médias Móveis
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()
        
        # Volume
        df['volume_change'] = df['volume'].pct_change()
        
        logger.info("Indicadores adicionados com sucesso")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao adicionar indicadores: {str(e)}")
        raise

def limpar_dados(df):
    """
    Limpa e prepara os dados para análise.
    
    Args:
        df (pd.DataFrame): DataFrame com dados e indicadores
        
    Returns:
        pd.DataFrame: DataFrame limpo e preparado
    """
    logger.info("Iniciando limpeza dos dados")
    
    try:
        # Remover linhas com valores NaN
        df_limpo = df.dropna()
        
        # Remover outliers
        for coluna in df_limpo.columns:
            if df_limpo[coluna].dtype in [np.float64, np.int64]:
                q1 = df_limpo[coluna].quantile(0.25)
                q3 = df_limpo[coluna].quantile(0.75)
                iqr = q3 - q1
                limite_inferior = q1 - 1.5 * iqr
                limite_superior = q3 + 1.5 * iqr
                df_limpo = df_limpo[
                    (df_limpo[coluna] >= limite_inferior) & 
                    (df_limpo[coluna] <= limite_superior)
                ]
        
        logger.info(f"Dados limpos: {len(df_limpo)} registros")
        return df_limpo
        
    except Exception as e:
        logger.error(f"Erro ao limpar dados: {str(e)}")
        raise

def preparar_dados_ml(df):
    """
    Prepara os dados para treinamento do modelo de machine learning.
    
    Args:
        df (pd.DataFrame): DataFrame com dados limpos
        
    Returns:
        tuple: (X, y) - Features e target
    """
    logger.info("Preparando dados para machine learning")
    
    try:
        # Criar target (retorno futuro)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Separar features e target
        features = df.drop(['target'], axis=1)
        target = df['target']
        
        # Remover última linha (target NaN)
        features = features[:-1]
        target = target[:-1]
        
        logger.info(f"Dados preparados: {len(features)} amostras")
        return features, target
        
    except Exception as e:
        logger.error(f"Erro ao preparar dados para ML: {str(e)}")
        raise

def salvar_csv(df, ticker):
    """
    Salva o DataFrame em um CSV com o nome do ativo.
    """
    os.makedirs("data", exist_ok=True)
    nome_arquivo = f"data/{ticker}_ativo_com_indicadores.csv"
    df.to_csv(nome_arquivo, index=True)
    print(f"💾 Dados salvos em: {nome_arquivo}")

# 🔁 Execução direta
if __name__ == "__main__":
    ticker_ativo = "AAPL"  # troque por "PETR4.SA", "BTC-USD", etc.
    dados = coletar_dados_15min(ticker_ativo)
    dados_com_indicadores = adicionar_indicadores(dados, ticker_ativo)
    salvar_csv(dados_com_indicadores, ticker_ativo)
