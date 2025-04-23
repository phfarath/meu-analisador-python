"""
Módulo responsável por aplicar filtros técnicos aos dados.
"""

import pandas as pd
import numpy as np
from logger import logger

def aplicar_filtros_tecnicos(df):
    """
        Aplica filtros técnicos para identificar momentos favoráveis para trades.
    
    Args:
        df (pd.DataFrame): DataFrame com dados e indicadores técnicos
        
    Returns:
        pd.DataFrame: DataFrame com filtros aplicados
    """
    logger.info("Aplicando filtros técnicos...")
    
    # Inicializa coluna de filtros
    df['filtros_ok'] = True
    
    # 1. Filtros de Tendência (mais restritivos)
    df['sma_20_trend'] = df['close'] > df['sma_20']
    df['ema_20_trend'] = df['close'] > df['ema_20']
    df['tendencia_ok'] = df['sma_20_trend'] | df['ema_20_trend']  # OR para mais flexibilidade
    
    # 2. Filtros de Momentum
    df['rsi_ok'] = (df['rsi'] > 25) & (df['rsi'] < 75)  # Ampliado para BTC
    df['macd_ok'] = df['macd'] > df['macd_signal']
    df['momentum_ok'] = df['rsi_ok'] | df['macd_ok']  # OR para mais flexibilidade
    
    # 3. Filtros de Volatilidade (mais restritivos)
    df['bb_ok'] = (df['close'] > df['bb_lower']) & (df['close'] < df['bb_upper'])
    df['atr_ok'] = df['atr'] > df['atr'].rolling(20).mean() * 0.4  # Reduzido threshold
    df['volatilidade_ok'] = df['bb_ok'] | df['atr_ok']  # OR para mais flexibilidade

    
    # 4. Filtros de Volume
    df['volume_ok'] = df['volume'] > df['volume'].rolling(20).mean() * 0.4  # Reduzido threshold
    df['obv_ok'] = df['obv'] > df['obv'].shift(1)
    df['volume_geral_ok'] = df['volume_ok'] | df['obv_ok']  # OR para mais flexibilidade
    
    # 5. Filtro de Força da Tendência
    df['adx_ok'] = df['adx'] > 15  # Reduzido para mais oportunidades
    
    # Combina todos os filtros (mais restritivo)
    df['filtros_ok'] = (
        df['tendencia_ok'] &
        (df['momentum_ok'] | df['volatilidade_ok']) &  # OR entre momentum e volatilidade
        df['volume_geral_ok'] &
        df['adx_ok']
    )
    
    # Debug: Calcula percentual de períodos que passam em cada filtro
    total_periodos = len(df)
    n_filtros_ok = df['filtros_ok'].sum()
    
    logger.info(f"Total de períodos analisados: {total_periodos}")
    logger.info(f"Períodos que passaram nos filtros: {n_filtros_ok} ({(n_filtros_ok/total_periodos*100):.1f}%)")
    logger.info(f"Tendência OK: {df['tendencia_ok'].sum()} ({(df['tendencia_ok'].sum()/total_periodos*100):.1f}%)")
    logger.info(f"Momentum OK: {df['momentum_ok'].sum()} ({(df['momentum_ok'].sum()/total_periodos*100):.1f}%)")
    logger.info(f"Volatilidade OK: {df['volatilidade_ok'].sum()} ({(df['volatilidade_ok'].sum()/total_periodos*100):.1f}%)")
    logger.info(f"Volume OK: {df['volume_geral_ok'].sum()} ({(df['volume_geral_ok'].sum()/total_periodos*100):.1f}%)")
    
    return df

