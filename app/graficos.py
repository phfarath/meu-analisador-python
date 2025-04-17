import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

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
