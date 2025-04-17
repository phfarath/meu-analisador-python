import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


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

    return {
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
