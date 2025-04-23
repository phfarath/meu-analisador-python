"""
Módulo para visualização de trades e resultados.
Implementa funções para criar visualizações interativas de operações e desempenho.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from logger import logger

def plot_trades(df, entradas, saidas, tipos_saida=None, title='Visualização de Trades'):
    """
    Cria um gráfico interativo mostrando os trades realizados.
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
        entradas (list): Lista de tuplas (data, preço) de entradas
        saidas (list): Lista de tuplas (data, preço) de saídas
        tipos_saida (list): Lista de tipos de saída (opcional)
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Figura com o gráfico
    """
    logger.info("Criando visualização de trades")
    
    try:
        # Criar figura
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3])
        
        # Adicionar candlesticks
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close'],
                                    name='Candles'),
                     row=1, col=1)
        
        # Adicionar volume
        fig.add_trace(go.Bar(x=df.index,
                            y=df['volume'],
                            name='Volume'),
                     row=2, col=1)
        
        # Adicionar pontos de entrada
        for entrada in entradas:
            fig.add_trace(go.Scatter(x=[entrada[0]],
                                    y=[entrada[1]],
                                    mode='markers',
                                    marker=dict(symbol='triangle-up',
                                              size=10,
                                              color='green'),
                                    name='Entrada'),
                         row=1, col=1)
        
        # Adicionar pontos de saída
        for i, saida in enumerate(saidas):
            cor = 'red'
            if tipos_saida and tipos_saida[i] == 'TP':
                cor = 'blue'
            elif tipos_saida and tipos_saida[i] == 'TS':
                cor = 'orange'
                
            fig.add_trace(go.Scatter(x=[saida[0]],
                                    y=[saida[1]],
                                    mode='markers',
                                    marker=dict(symbol='triangle-down',
                                              size=10,
                                              color=cor),
                                    name='Saída'),
                         row=1, col=1)
        
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title='Data',
            yaxis_title='Preço',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        logger.info("Visualização de trades criada com sucesso")
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar visualização de trades: {str(e)}")
        raise

def plot_retornos(retornos, title='Distribuição de Retornos'):
    """
    Cria um gráfico de distribuição dos retornos dos trades.
    
    Args:
        retornos (list): Lista de retornos dos trades
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Figura com o gráfico
    """
    logger.info("Criando visualização de distribuição de retornos")
    
    try:
        # Criar figura
        fig = go.Figure()
        
        # Adicionar histograma
        fig.add_trace(go.Histogram(x=retornos,
                                  nbinsx=50,
                                  name='Retornos'))
        
        # Adicionar linha de média
        media = np.mean(retornos)
        fig.add_vline(x=media,
                      line_dash="dash",
                      line_color="red",
                      annotation_text=f"Média: {media:.4f}")
        
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title='Retorno',
            yaxis_title='Frequência',
            showlegend=False
        )
        
        logger.info("Visualização de distribuição de retornos criada com sucesso")
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar visualização de distribuição de retornos: {str(e)}")
        raise

def plot_equity_curve(df_trades, capital_inicial=10000, title='Curva de Equity'):
    """
    Cria um gráfico da curva de equity ao longo do tempo.
    
    Args:
        df_trades (pd.DataFrame): DataFrame com informações dos trades
        capital_inicial (float): Capital inicial
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Figura com o gráfico
    """
    logger.info("Criando visualização da curva de equity")
    
    try:
        # Calcular equity ao longo do tempo
        df_trades = df_trades.sort_values('data')
        df_trades['equity'] = capital_inicial + df_trades['resultado'].cumsum()
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar linha de equity
        fig.add_trace(go.Scatter(x=df_trades['data'],
                                y=df_trades['equity'],
                                mode='lines',
                                name='Equity'))
        
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title='Data',
            yaxis_title='Equity',
            showlegend=False
        )
        
        logger.info("Visualização da curva de equity criada com sucesso")
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar visualização da curva de equity: {str(e)}")
        raise

def plot_drawdown(df_trades, capital_inicial=10000, title='Drawdown'):
    """
    Cria um gráfico do drawdown ao longo do tempo.
    
    Args:
        df_trades (pd.DataFrame): DataFrame com informações dos trades
        capital_inicial (float): Capital inicial
        title (str): Título do gráfico
        
    Returns:
        plotly.graph_objects.Figure: Figura com o gráfico
    """
    logger.info("Criando visualização do drawdown")
    
    try:
        # Calcular drawdown
        df_trades = df_trades.sort_values('data')
        df_trades['equity'] = capital_inicial + df_trades['resultado'].cumsum()
        df_trades['max_equity'] = df_trades['equity'].cummax()
        df_trades['drawdown'] = (df_trades['equity'] - df_trades['max_equity']) / df_trades['max_equity']
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar área de drawdown
        fig.add_trace(go.Scatter(x=df_trades['data'],
                                y=df_trades['drawdown'],
                                fill='tozeroy',
                                name='Drawdown'))
        
        # Configurar layout
        fig.update_layout(
            title=title,
            xaxis_title='Data',
            yaxis_title='Drawdown',
            showlegend=False
        )
        
        logger.info("Visualização do drawdown criada com sucesso")
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar visualização do drawdown: {str(e)}")
        raise
