"""
Módulo para visualização de dados e resultados.
Implementa funções para criação de gráficos e visualizações de análise técnica.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logger import logger

def plotar_candle_rsi_macd(df, ticker="Ativo"):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(f"{ticker} - Candlestick",
                                        "RSI (Índice de Força Relativa)",
                                        "MACD"))

    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df["open"],
                                 high=df["high"],
                                 low=df["low"],
                                 close=df["close"],
                                 name="Candlestick"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], line=dict(color="blue"), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], line=dict(color="orange"), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], line=dict(color="purple"), name="Sinal"), row=3, col=1)

    fig.update_layout(height=800, width=1000, title_text=f"Análise Técnica: {ticker}",
                      xaxis_rangeslider_visible=False)
    fig.show()

def carregar_e_plotar(ticker="AAPL"):
    nome_arquivo = f"data/{ticker}_ativo_com_indicadores.csv"
    
    if not os.path.exists(nome_arquivo):
        print(f"❌ Arquivo {nome_arquivo} não encontrado.")
        return
    
    df = pd.read_csv(nome_arquivo)

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
    else:
        print("⚠️ Coluna 'Datetime' não encontrada. Verifique o CSV.")
        return

    plotar_candle_rsi_macd(df, ticker)

# 🔁 Exemplo de uso direto
if __name__ == "__main__":
    carregar_e_plotar("AAPL")  # Troque por "PETR4.SA", "BTC-USD", etc.

def plot_candlestick(df, title='Gráfico de Candles'):
    """
    Cria um gráfico de candlestick com indicadores técnicos.
    
    Args:
        df (pd.DataFrame): DataFrame com dados OHLCV
        title (str): Título do gráfico
        
    Returns:
        go.Figure: Figura Plotly com o gráfico
    """
    logger.info("Criando gráfico de candlestick")
    
    try:
        # Criar figura com subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Adicionar candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Candles'
            ),
            row=1, col=1
        )
        
        # Adicionar volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Adicionar RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI'
            ),
            row=3, col=1
        )
        
        # Atualizar layout
        fig.update_layout(
            title=title,
            yaxis_title='Preço',
            yaxis2_title='Volume',
            yaxis3_title='RSI',
            xaxis_rangeslider_visible=False
        )
        
        logger.info("Gráfico criado com sucesso")
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar gráfico: {str(e)}")
        raise

def plot_correlacao(df):
    """
    Cria uma matriz de correlação entre as features.
    
    Args:
        df (pd.DataFrame): DataFrame com features
        
    Returns:
        plt.Figure: Figura Matplotlib com a matriz de correlação
    """
    logger.info("Criando matriz de correlação")
    
    try:
        # Calcular correlação
        corr = df.corr()
        
        # Criar figura
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f'
        )
        plt.title('Matriz de Correlação')
        
        logger.info("Matriz de correlação criada com sucesso")
        return plt.gcf()
        
    except Exception as e:
        logger.error(f"Erro ao criar matriz de correlação: {str(e)}")
        raise

def plot_distribuicao(df, coluna):
    """
    Cria um gráfico de distribuição para uma coluna específica.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        coluna (str): Nome da coluna para análise
        
    Returns:
        plt.Figure: Figura Matplotlib com o gráfico de distribuição
    """
    logger.info(f"Criando gráfico de distribuição para {coluna}")
    
    try:
        # Criar figura
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df,
            x=coluna,
            kde=True
        )
        plt.title(f'Distribuição de {coluna}')
        
        logger.info("Gráfico de distribuição criado com sucesso")
        return plt.gcf()
        
    except Exception as e:
        logger.error(f"Erro ao criar gráfico de distribuição: {str(e)}")
        raise

def plot_series_temporais(df, colunas):
    """
    Cria gráficos de séries temporais para múltiplas colunas.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        colunas (list): Lista de colunas para plotar
        
    Returns:
        plt.Figure: Figura Matplotlib com os gráficos
    """
    logger.info("Criando gráficos de séries temporais")
    
    try:
        # Criar figura
        fig, axes = plt.subplots(len(colunas), 1, figsize=(12, 4*len(colunas)))
        
        # Plotar cada coluna
        for i, coluna in enumerate(colunas):
            axes[i].plot(df.index, df[coluna])
            axes[i].set_title(coluna)
            axes[i].grid(True)
        
        plt.tight_layout()
        
        logger.info("Gráficos de séries temporais criados com sucesso")
        return fig
        
    except Exception as e:
        logger.error(f"Erro ao criar gráficos de séries temporais: {str(e)}")
        raise

def salvar_grafico(fig, nome_arquivo, formato='png'):
    """
    Salva um gráfico em arquivo.
    
    Args:
        fig: Figura do gráfico (Matplotlib ou Plotly)
        nome_arquivo (str): Nome do arquivo
        formato (str): Formato do arquivo ('png', 'jpg', 'html')
    """
    logger.info(f"Salvando gráfico em {nome_arquivo}")
    
    try:
        if isinstance(fig, go.Figure):
            if formato == 'html':
                fig.write_html(nome_arquivo)
            else:
                fig.write_image(nome_arquivo)
        else:
            fig.savefig(nome_arquivo, format=formato, dpi=300)
        
        logger.info("Gráfico salvo com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao salvar gráfico: {str(e)}")
        raise
