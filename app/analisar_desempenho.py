"""
Módulo para análise de desempenho do sistema de trading.
Implementa funções para cálculo de métricas e análise de resultados.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
from logger import logger


def calculo_desempenho(retornos, tipos_saida, limiar=1e-6):
    """Analisa o desempenho da estratégia com métricas avançadas"""

    if len(retornos) != len(tipos_saida):
        raise ValueError("⚠️ 'retornos' e 'tipos_saida' devem ter o mesmo tamanho.")

    df_resultados = pd.DataFrame({
        'retorno': [float(r) for r in retornos],
        'tipo_saida': tipos_saida
    })

    df_validos = df_resultados[np.abs(df_resultados['retorno']) > limiar]
    total_trades = len(df_validos)

    if total_trades == 0:
        return {'total_trades': 0}

    trades_ganhos = df_validos[df_validos['retorno'] > 0]
    trades_perdas = df_validos[df_validos['retorno'] < 0]

    win_rate = len(trades_ganhos) / total_trades
    avg_gain = trades_ganhos['retorno'].mean() if not trades_ganhos.empty else 0
    avg_loss = trades_perdas['retorno'].mean() if not trades_perdas.empty else 0
    profit_factor = abs(trades_ganhos['retorno'].sum() / trades_perdas['retorno'].sum()) if not trades_perdas.empty else float('inf')

    # Curva de capital
    capital_acumulado = [10000]
    for r in df_validos['retorno']:
        capital_acumulado.append(capital_acumulado[-1] * (1 + r))
    capital_acumulado = np.array(capital_acumulado)

    # Max drawdown
    peak = np.maximum.accumulate(capital_acumulado)
    drawdowns = (peak - capital_acumulado) / peak
    max_drawdown = np.max(drawdowns)

    # Sharpe Ratio
    retornos_log = np.log1p(df_validos['retorno'])
    sharpe = np.mean(retornos_log) / np.std(retornos_log) * np.sqrt(252) if np.std(retornos_log) > 0 else 0

    # Sortino Ratio
    downside_risk = np.std(retornos_log[retornos_log < 0])
    sortino = np.mean(retornos_log) / downside_risk * np.sqrt(252) if downside_risk > 0 else 0

    # Expectancy por tipo
    expectancy_tipo = df_validos.groupby('tipo_saida')['retorno'].mean().to_dict()

    # Matriz de confusão simples (contagem por tipo)
    tipo_labels = sorted(df_validos['tipo_saida'].unique())
    matriz_confusao = dict(zip(tipo_labels, df_validos['tipo_saida'].value_counts().to_dict().items()))

    metricas = {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'expectancy': (win_rate * avg_gain) - ((1 - win_rate) * abs(avg_loss)),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'expectancy_por_tipo': expectancy_tipo,
        'matriz_tipos': matriz_confusao
    }
    
    # Gerar relatório amigável
    gerar_relatorio_amigavel(metricas)
    
    return metricas

def calcular_metricas(df_trades):
    """
    Calcula métricas de desempenho a partir dos trades realizados.
    
    Args:
        df_trades (pd.DataFrame): DataFrame com informações dos trades
        
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    logger.info("Calculando métricas de desempenho")
    
    try:
        # Inicializar métricas
        metricas = {
            'total_trades': len(df_trades),
            'trades_lucrativos': 0,
            'trades_prejuizo': 0,
            'lucro_total': 0,
            'prejuizo_total': 0,
            'maior_lucro': 0,
            'maior_prejuizo': 0,
            'sequencia_vitorias': 0,
            'sequencia_derrotas': 0
        }
        
        # Calcular métricas básicas
        for _, trade in df_trades.iterrows():
            resultado = trade['preco_saida'] - trade['preco_entrada']
            
            if resultado > 0:
                metricas['trades_lucrativos'] += 1
                metricas['lucro_total'] += resultado
                metricas['maior_lucro'] = max(metricas['maior_lucro'], resultado)
                metricas['sequencia_vitorias'] += 1
                metricas['sequencia_derrotas'] = 0
            else:
                metricas['trades_prejuizo'] += 1
                metricas['prejuizo_total'] += abs(resultado)
                metricas['maior_prejuizo'] = min(metricas['maior_prejuizo'], resultado)
                metricas['sequencia_derrotas'] += 1
                metricas['sequencia_vitorias'] = 0
        
        # Calcular métricas derivadas
        metricas['win_rate'] = metricas['trades_lucrativos'] / metricas['total_trades']
        metricas['profit_factor'] = metricas['lucro_total'] / metricas['prejuizo_total'] if metricas['prejuizo_total'] > 0 else float('inf')
        metricas['retorno_total'] = metricas['lucro_total'] - metricas['prejuizo_total']
        metricas['retorno_medio'] = metricas['retorno_total'] / metricas['total_trades']
        
        logger.info("Métricas calculadas com sucesso")
        return metricas
        
    except Exception as e:
        logger.error(f"Erro ao calcular métricas: {str(e)}")
        raise

def analisar_drawdown(df_trades):
    """
    Analisa o drawdown do capital ao longo do tempo.
    
    Args:
        df_trades (pd.DataFrame): DataFrame com informações dos trades
        
    Returns:
        dict: Dicionário com informações de drawdown
    """
    logger.info("Analisando drawdown")
    
    try:
        # Calcular evolução do capital
        df_trades['capital_acumulado'] = df_trades['resultado'].cumsum()
        
        # Calcular drawdown
        df_trades['max_capital'] = df_trades['capital_acumulado'].cummax()
        df_trades['drawdown'] = df_trades['capital_acumulado'] - df_trades['max_capital']
        
        # Calcular métricas de drawdown
        drawdown_info = {
            'max_drawdown': df_trades['drawdown'].min(),
            'drawdown_atual': df_trades['drawdown'].iloc[-1],
            'tempo_max_drawdown': df_trades.loc[df_trades['drawdown'].idxmin(), 'data']
        }
        
        logger.info("Análise de drawdown concluída")
        return drawdown_info
        
    except Exception as e:
        logger.error(f"Erro ao analisar drawdown: {str(e)}")
        raise

def analisar_risco(df_trades, capital_inicial):
    """
    Analisa métricas de risco do sistema.
    
    Args:
        df_trades (pd.DataFrame): DataFrame com informações dos trades
        capital_inicial (float): Capital inicial do sistema
        
    Returns:
        dict: Dicionário com métricas de risco
    """
    logger.info("Analisando métricas de risco")
    
    try:
        # Calcular retornos
        df_trades['retorno'] = df_trades['resultado'] / capital_inicial
        
        # Calcular métricas de risco
        risco_info = {
            'volatilidade': df_trades['retorno'].std(),
            'sharpe_ratio': df_trades['retorno'].mean() / df_trades['retorno'].std() if df_trades['retorno'].std() > 0 else 0,
            'var_95': df_trades['retorno'].quantile(0.05),
            'max_loss_diario': df_trades.groupby(df_trades['data'].dt.date)['resultado'].sum().min()
        }
        
        logger.info("Análise de risco concluída")
        return risco_info
        
    except Exception as e:
        logger.error(f"Erro ao analisar risco: {str(e)}")
        raise

def gerar_relatorio(metricas, drawdown_info, risco_info):
    """
    Gera um relatório completo de desempenho.
    
    Args:
        metricas (dict): Métricas básicas de desempenho
        drawdown_info (dict): Informações de drawdown
        risco_info (dict): Métricas de risco
        
    Returns:
        str: Relatório formatado
    """
    logger.info("Gerando relatório de desempenho")
    
    try:
        relatorio = """
        RELATÓRIO DE DESEMPENHO
        ======================
        
        MÉTRICAS BÁSICAS
        ---------------
        Total de Trades: {total_trades}
        Trades Lucrativos: {trades_lucrativos} ({win_rate:.2%})
        Lucro Total: R$ {lucro_total:.2f}
        Prejuízo Total: R$ {prejuizo_total:.2f}
        Retorno Total: R$ {retorno_total:.2f}
        Retorno Médio por Trade: R$ {retorno_medio:.2f}
        Profit Factor: {profit_factor:.2f}
        
        DRAWDOWN
        --------
        Máximo Drawdown: R$ {max_drawdown:.2f}
        Drawdown Atual: R$ {drawdown_atual:.2f}
        Data do Máximo Drawdown: {tempo_max_drawdown}
        
        RISCO
        -----
        Volatilidade: {volatilidade:.2%}
        Sharpe Ratio: {sharpe_ratio:.2f}
        VaR 95%: {var_95:.2%}
        Máxima Perda Diária: R$ {max_loss_diario:.2f}
        """.format(
            **metricas,
            **drawdown_info,
            **risco_info
        )
        
        logger.info("Relatório gerado com sucesso")
        return relatorio
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {str(e)}")
        raise

def gerar_relatorio_amigavel(metricas):
    """
    Gera um relatório amigável com as métricas de desempenho.
    """
    logger.info("\n" + "="*50)
    logger.info("📊 RELATÓRIO DE DESEMPENHO")
    logger.info("="*50)
    
    # Resumo Geral
    logger.info("\n🎯 RESUMO GERAL:")
    logger.info(f"Total de operações realizadas: {metricas['total_trades']}")
    logger.info(f"Taxa de acerto: {metricas['win_rate']*100:.1f}%")
    
    # Resultados Financeiros
    logger.info("\n💰 RESULTADOS FINANCEIROS:")
    logger.info(f"Lucro médio por operação: R$ {metricas['avg_gain']:,.2f}")
    logger.info(f"Prejuízo médio por operação: R$ {abs(metricas['avg_loss']):,.2f}")
    logger.info(f"Expectativa de lucro por operação: R$ {metricas['expectancy']:,.2f}")
    
    # Análise de Risco
    logger.info("\n⚠️ ANÁLISE DE RISCO:")
    logger.info(f"Maior queda do capital (Drawdown): R$ {abs(metricas['max_drawdown']):,.2f}")
    logger.info(f"Relação risco/retorno (Profit Factor): {metricas['profit_factor']:.2f}")
    
    # Detalhamento por Tipo de Saída
    logger.info("\n🎯 DETALHAMENTO POR TIPO DE SAÍDA:")
    for tipo, resultado in metricas['expectancy_por_tipo'].items():
        if tipo == 'take_profit':
            logger.info(f"Ganho médio no Take Profit: R$ {resultado:,.2f}")
        elif tipo == 'stop_loss':
            logger.info(f"Perda média no Stop Loss: R$ {abs(resultado):,.2f}")
    
    # Avaliação Final
    logger.info("\n🏆 AVALIAÇÃO FINAL:")
    if metricas['win_rate'] > 0.5 and metricas['profit_factor'] > 2:
        logger.info("EXCELENTE! O sistema está apresentando resultados muito promissores.")
    elif metricas['win_rate'] > 0.4 and metricas['profit_factor'] > 1.5:
        logger.info("BOM! O sistema está lucrativo, mas há espaço para melhorias.")
    else:
        logger.info("ATENÇÃO! O sistema precisa de ajustes para melhorar o desempenho.")
    
    logger.info("="*50)
