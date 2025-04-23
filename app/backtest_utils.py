"""
Módulo de utilitários para backtest.
Implementa funções para simulação e análise de estratégias de trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from logger import logger

def backtest_avancado(df, probs, capital_inicial=10000, 
                     stop_loss_pct=0.02, take_profit_pct=0.04,
                     comissao=0.001, slippage=0.0005):
    """
    Executa um backtest avançado com múltiplas métricas.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        probs (np.array): Probabilidades de previsão
        capital_inicial (float): Capital inicial
        stop_loss_pct (float): Porcentagem para stop loss
        take_profit_pct (float): Porcentagem para take profit
        comissao (float): Comissão por operação
        slippage (float): Deslizamento de preço
        
    Returns:
        tuple: (métricas, DataFrame de trades, entradas, saídas)
    """
    logger.info("Iniciando backtest avançado")
    
    try:
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
        
        # Iterar sobre os dados
        for i in range(len(df)):
            preco_atual = df['close'].iloc[i]
            
            # Verificar stop loss e take profit
            if posicao_aberta:
                if preco_atual <= stop_loss:
                    # Stop loss atingido
                    resultado = (stop_loss - preco_entrada) * (1 - comissao - slippage)
                    capital += resultado
                    trades.append({
                        'data': df.index[i],
                        'tipo': 'venda',
                        'preco_entrada': preco_entrada,
                        'preco_saida': stop_loss,
                        'resultado': resultado
                    })
                    saidas.append((df.index[i], stop_loss))
                    posicao_aberta = False
                    
                elif preco_atual >= take_profit:
                    # Take profit atingido
                    resultado = (take_profit - preco_entrada) * (1 - comissao - slippage)
                    capital += resultado
                    trades.append({
                        'data': df.index[i],
                        'tipo': 'venda',
                        'preco_entrada': preco_entrada,
                        'preco_saida': take_profit,
                        'resultado': resultado
                    })
                    saidas.append((df.index[i], take_profit))
                    posicao_aberta = False

            # Verificar sinais de entrada
            if not posicao_aberta and probs[i] > 0.6:
                # Entrar na posição
                preco_entrada = preco_atual * (1 + slippage)
                stop_loss = preco_entrada * (1 - stop_loss_pct)
                take_profit = preco_entrada * (1 + take_profit_pct)
                posicao_aberta = True
                entradas.append((df.index[i], preco_entrada))
        
        # Converter trades para DataFrame
        df_trades = pd.DataFrame(trades)
        
        # Calcular métricas
        metricas = calcular_metricas(df_trades)
        metricas['capital_final'] = capital
        metricas['retorno_total'] = (capital - capital_inicial) / capital_inicial
        
        logger.info("Backtest concluído com sucesso")
        return metricas, df_trades, entradas, saidas
        
    except Exception as e:
        logger.error(f"Erro durante o backtest: {str(e)}")
        raise

def otimizar_parametros(df, probs, parametros):
    """
    Otimiza os parâmetros do backtest.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        probs (np.array): Probabilidades de previsão
        parametros (dict): Dicionário com parâmetros para otimização
        
    Returns:
        dict: Melhores parâmetros encontrados
    """
    logger.info("Iniciando otimização de parâmetros")
    
    try:
        melhor_retorno = -float('inf')
        melhores_parametros = None
        
        # Testar diferentes combinações de parâmetros
        for sl in parametros['stop_loss']:
            for tp in parametros['take_profit']:
                for com in parametros['comissao']:
                    metricas, _, _, _ = backtest_avancado(
                        df, probs,
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        comissao=com
                    )
                    
                    if metricas['retorno_total'] > melhor_retorno:
                        melhor_retorno = metricas['retorno_total']
                        melhores_parametros = {
                            'stop_loss': sl,
                            'take_profit': tp,
                            'comissao': com
                        }
        
        logger.info("Otimização concluída")
        return melhores_parametros
        
    except Exception as e:
        logger.error(f"Erro durante a otimização: {str(e)}")
        raise

def analisar_robustez(df, probs, n_simulacoes=100):
    """
    Analisa a robustez da estratégia através de simulações.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        probs (np.array): Probabilidades de previsão
        n_simulacoes (int): Número de simulações
        
    Returns:
        dict: Métricas de robustez
    """
    logger.info("Iniciando análise de robustez")
    
    try:
        retornos = []
        win_rates = []
        profit_factors = []
        
        # Executar múltiplas simulações
        for _ in range(n_simulacoes):
            # Embaralhar dados mantendo a estrutura temporal
            df_simulado = df.copy()
            df_simulado['probs'] = probs
            df_simulado = df_simulado.sample(frac=1).sort_index()
            
            # Executar backtest
            metricas, _, _, _ = backtest_avancado(
                df_simulado,
                df_simulado['probs'].values
            )
            
            retornos.append(metricas['retorno_total'])
            win_rates.append(metricas['win_rate'])
            profit_factors.append(metricas['profit_factor'])
        
        # Calcular métricas de robustez
        robustez = {
            'retorno_medio': np.mean(retornos),
            'retorno_std': np.std(retornos),
            'win_rate_medio': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'profit_factor_medio': np.mean(profit_factors),
            'profit_factor_std': np.std(profit_factors)
        }
        
        logger.info("Análise de robustez concluída")
        return robustez
        
    except Exception as e:
        logger.error(f"Erro durante a análise de robustez: {str(e)}")
        raise

