import pandas as pd
import ta

def aplicar_filtros_mercado(df):
    df = df.copy()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['tendencia'] = df['close'] > df['sma_50']

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['volatilidade_alta'] = df['atr'] > df['atr'].rolling(20).mean() * 1.5

    df['volume_adequado'] = df['volume'] > df['volume'].rolling(20).mean() * 0.7

    if "Datetime" in df.columns:
        df["hora"] = pd.to_datetime(df["Datetime"]).dt.hour
    else:
        df["hora"] = df.index.hour
    df["horario_favoravel"] = df["hora"].between(10, 16)

    df["filtros_ok"] = (
        df["tendencia"] &
        ~df["volatilidade_alta"] &
        df["volume_adequado"] &
        df["horario_favoravel"]
    )
    return df
