"""
Módulo de backtest com estratégia agressiva.
Implementa funções para simulação de estratégias de trading com maior risco.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from logger import logger
from analisar_desempenho import calculo_desempenho

def backtest_agressivo(df, probs, capital_inicial=10000,
                      stop_loss_pct=0.015, take_profit_pct=0.045,
                      comissao=0.001, slippage=0.0005,
                      alavancagem=2.0, trailing_stop=True,
                      trailing_stop_offset=0.005):
    """
    Executa um backtest com estratégia agressiva e trailing stop.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        probs (np.array): Probabilidades de previsão
        capital_inicial (float): Capital inicial
        stop_loss_pct (float): Porcentagem para stop loss
        take_profit_pct (float): Porcentagem para take profit
        comissao (float): Comissão por operação
        slippage (float): Deslizamento de preço
        alavancagem (float): Nível de alavancagem
        trailing_stop (bool): Usar trailing stop
        trailing_stop_offset (float): Offset do trailing stop
        
    Returns:
        tuple: (retornos, entradas, saídas, tipos_saida)
    """
    logger.info("Iniciando backtest agressivo")
    
    try:
        # Ajustar parâmetros para estratégia agressiva
        capital_efetivo = capital_inicial * alavancagem
        stop_loss_pct = stop_loss_pct / alavancagem
        take_profit_pct = take_profit_pct / alavancagem
        
        # Inicializar variáveis
        capital = capital_inicial
        posicao_aberta = False
        preco_entrada = 0
        stop_loss = 0
        take_profit = 0
        trailing_stop_price = 0
        max_price_since_entry = 0
        
        # Listas para armazenar trades
        trades = []
        entradas = []
        saidas = []
        tipos_saida = []
        
        # Iterar sobre os dados
        for i in range(len(df)):
            preco_atual = df['close'].iloc[i]
            
            # Verificar stop loss, take profit e trailing stop
            if posicao_aberta:
                # Atualizar preço máximo desde a entrada
                if preco_atual > max_price_since_entry:
                    max_price_since_entry = preco_atual
                    if trailing_stop:
                        # Atualizar trailing stop
                        novo_stop = max_price_since_entry * (1 - trailing_stop_offset)
                        if novo_stop > stop_loss:
                            stop_loss = novo_stop
                
                if preco_atual <= stop_loss:
                    # Stop loss ou trailing stop atingido
                    resultado = (stop_loss - preco_entrada) * (1 - comissao - slippage) * alavancagem
                    capital += resultado
                    trades.append({
                        'data': df.index[i],
                        'tipo': 'venda',
                        'preco_entrada': preco_entrada,
                        'preco_saida': stop_loss,
                        'resultado': resultado,
                        'alavancagem': alavancagem
                    })
                    saidas.append((df.index[i], stop_loss))
                    tipos_saida.append('stop_loss')
                    posicao_aberta = False
                    
                elif preco_atual >= take_profit:
                    # Take profit atingido
                    resultado = (take_profit - preco_entrada) * (1 - comissao - slippage) * alavancagem
                    capital += resultado
                    trades.append({
                        'data': df.index[i],
                        'tipo': 'venda',
                        'preco_entrada': preco_entrada,
                        'preco_saida': take_profit,
                        'resultado': resultado,
                        'alavancagem': alavancagem
                    })
                    saidas.append((df.index[i], take_profit))
                    tipos_saida.append('take_profit')
                    posicao_aberta = False

            # Verificar sinais de entrada com filtros mais rigorosos
            if not posicao_aberta and probs[i] > 0.60 and df['filtros_ok'].iloc[i]:
                # Verificar tendência adicional
                if (df['close'].iloc[i] > df['sma_20'].iloc[i] or
                    df['macd'].iloc[i] > df['macd_signal'].iloc[i] or
                    30 < df['rsi'].iloc[i] < 70):
                    
                    # Entrar na posição
                    preco_entrada = preco_atual * (1 + slippage)
                    stop_loss = preco_entrada * (1 - stop_loss_pct)
                    take_profit = preco_entrada * (1 + take_profit_pct)
                    max_price_since_entry = preco_entrada
                    trailing_stop_price = stop_loss
                    posicao_aberta = True
                    entradas.append((df.index[i], preco_entrada))
        
        # Calcular métricas
        metricas = calculo_desempenho([t['resultado'] for t in trades], tipos_saida)
        metricas['capital_final'] = capital
        metricas['retorno_total'] = (capital - capital_inicial) / capital_inicial
        metricas['alavancagem'] = alavancagem
        
        logger.info("Backtest agressivo concluído")
        return [t['resultado'] for t in trades], entradas, saidas, tipos_saida
        
    except Exception as e:
        logger.error(f"Erro durante o backtest agressivo: {str(e)}")
        raise

def backtest_super_agressivo(df, probs, capital_inicial=10000,
                           stop_loss_pct=0.05, take_profit_pct=0.10,
                           comissao=0.001, slippage=0.0005,
                           alavancagem=5.0):
    """
    Executa um backtest com estratégia super agressiva.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        probs (np.array): Probabilidades de previsão
        capital_inicial (float): Capital inicial
        stop_loss_pct (float): Porcentagem para stop loss
        take_profit_pct (float): Porcentagem para take profit
        comissao (float): Comissão por operação
        slippage (float): Deslizamento de preço
        alavancagem (float): Nível de alavancagem
        
    Returns:
        tuple: (métricas, DataFrame de trades, entradas, saídas)
    """
    logger.info("Iniciando backtest super agressivo")
    
    try:
        # Ajustar parâmetros para estratégia super agressiva
        capital_efetivo = capital_inicial * alavancagem
        stop_loss_pct = stop_loss_pct / alavancagem
        take_profit_pct = take_profit_pct / alavancagem
        
        # Inicializar variáveis
        capital = capital_inicial
        posicao_aberta = False
        preco_entrada = 0
        stop_loss = 0
        take_profit = 0
        
        # Listas para armazenar trades
        trades = []
        entradas = []
        saidas = []
        tipos_saida = []
        
        # Iterar sobre os dados
        for i in range(len(df)):
            preco_atual = df['close'].iloc[i]
            
            # Verificar stop loss e take profit
            if posicao_aberta:
                if preco_atual <= stop_loss:
                    # Stop loss atingido
                    resultado = (stop_loss - preco_entrada) * (1 - comissao - slippage) * alavancagem
                    capital += resultado
                    trades.append({
                        'data': df.index[i],
                        'tipo': 'venda',
                        'preco_entrada': preco_entrada,
                        'preco_saida': stop_loss,
                        'resultado': resultado,
                        'alavancagem': alavancagem
                    })
                    saidas.append((df.index[i], stop_loss))
                    tipos_saida.append('stop_loss')
                    posicao_aberta = False
                    
                elif preco_atual >= take_profit:
                    # Take profit atingido
                    resultado = (take_profit - preco_entrada) * (1 - comissao - slippage) * alavancagem
                    capital += resultado
                    trades.append({
                        'data': df.index[i],
                        'tipo': 'venda',
                        'preco_entrada': preco_entrada,
                        'preco_saida': take_profit,
                        'resultado': resultado,
                        'alavancagem': alavancagem
                    })
                    saidas.append((df.index[i], take_profit))
                    tipos_saida.append('take_profit')
                    posicao_aberta = False
            
            # Verificar sinais de entrada (critérios ainda mais flexíveis)
            if not posicao_aberta and probs[i] > 0.5:  # Limiar mais baixo ainda
                # Entrar na posição
                preco_entrada = preco_atual * (1 + slippage)
                stop_loss = preco_entrada * (1 - stop_loss_pct)
                take_profit = preco_entrada * (1 + take_profit_pct)
                posicao_aberta = True
                entradas.append((df.index[i], preco_entrada))
        
        # Converter trades para DataFrame
        df_trades = pd.DataFrame(trades)
        
        # Calcular métricas
        metricas = calculo_desempenho([t['resultado'] for t in trades], tipos_saida)
        metricas['capital_final'] = capital
        metricas['retorno_total'] = (capital - capital_inicial) / capital_inicial
        metricas['alavancagem'] = alavancagem
        
        logger.info("Backtest super agressivo concluído")
        return [t['resultado'] for t in trades], entradas, saidas, tipos_saida
        
    except Exception as e:
        logger.error(f"Erro durante o backtest super agressivo: {str(e)}")
        raise