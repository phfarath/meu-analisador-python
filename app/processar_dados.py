import yfinance as yf
import pandas as pd
import ta
import os

def coletar_dados_15min(ticker, dias=60):
    """
    Coleta dados de 15 em 15 minutos de um ativo usando Yahoo Finance.
    """
    print(f"⏳ Coletando dados do ativo {ticker}...")
    df = yf.download(
        tickers=str(ticker),  # garante que seja string pura
        interval="15m",
        period=f"{dias}d",
        auto_adjust=False,
        progress=False
    )
    df.dropna(inplace=True)

    # 🔧 Se MultiIndex, achata
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("✅ Coleta concluída.")
    return df

def adicionar_indicadores(df, ticker):
    """
    Adiciona indicadores técnicos ao DataFrame de preços.
    """
    df = df.copy()

    # Renomear colunas para facilitar
    df.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    print("🔍 Colunas disponíveis:", df.columns.tolist())

    # Garantir que colunas numéricas estão no formato correto
    colunas_esperadas = ["open", "high", "low", "close", "volume"]
    for col in colunas_esperadas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"⚠️ Coluna '{col}' não encontrada no DataFrame!")

    # Indicadores técnicos
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    macd = ta.trend.MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["sma_20"] = ta.trend.SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(close=df["close"], window=20).ema_indicator()

    bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # Adicionar o nome do ativo
    df["ticker"] = ticker

    df.dropna(inplace=True)
    print("✅ Indicadores adicionados.")
    return df

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
