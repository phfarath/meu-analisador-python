import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plotar_trades_completos(df_test, entradas, saidas, retornos=None):
    fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # --- Subplot 1: Tentativas de trade (sinais e filtros) ---
    ax1 = axs[0]
    ax1.plot(df_test.index, df_test['close'], label='Preço (close)', color='gray', linewidth=1)

    sinais = df_test[df_test['sinal_compra'] == 1]
    ax1.scatter(sinais.index, sinais['close'], label='Sinais de Compra', color='blue', marker='^', s=50)

    filtros = df_test[df_test['filtros_ok']]
    ax1.scatter(filtros.index, filtros['close'], label='Filtro OK', color='green', marker='o', s=40, alpha=0.6)

    validos = df_test[(df_test['sinal_compra'] == 1) & (df_test['filtros_ok'])]
    ax1.scatter(validos.index, validos['close'], label='Sinal + Filtro', color='red', marker='*', s=130)

    ax1.set_title('🔍 Tentativas de Trade (Sinais e Filtros)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Preço')
    ax1.legend()
    ax1.grid(True)

    # --- Subplot 2: Trades Executados (entradas e saídas) ---
    ax2 = axs[1]
    ax2.plot(df_test.index, df_test['close'], label='Preço (close)', alpha=0.8, linewidth=1)

    entrada_times = [e[0] for e in entradas]
    entrada_precos = [e[1] for e in entradas]
    saida_times = [s[0] for s in saidas]
    saida_precos = [s[1] for s in saidas]

    ax2.scatter(entrada_times, entrada_precos, color='green', label='Entradas', marker='^', s=100)
    ax2.scatter(saida_times, saida_precos, color='red', label='Saídas', marker='v', s=100)

    ax2.set_title('✅ Trades Executados', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tempo')
    ax2.set_ylabel('Preço')
    ax2.legend()
    ax2.grid(True)

    # --- Subplot 3: Curva de Capital ---
    ax3 = axs[2]
    if retornos is not None:
        capital = [10000]
        for r in retornos:
            if r != 0:
                capital.append(capital[-1] * (1 + r))
        ax3.plot(capital, label='Curva de Capital', color='blue')
        ax3.set_title('📈 Evolução do Capital', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Capital (R$)')
        ax3.set_xlabel('Número de Trades')
        ax3.grid(True)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Nenhum retorno para plotar.', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('📈 Curva de Capital', fontsize=14, fontweight='bold')
        ax3.grid(True)

    # --- Subplot 4: Heatmap de RSI ---
    ax4 = axs[3]
    rsi = df_test['rsi'].fillna(0).clip(0, 100)
    rsi_colors = np.where(rsi > 70, 'red', np.where(rsi < 30, 'green', 'yellow'))
    ax4.bar(df_test.index, rsi, color=rsi_colors, width=0.01)
    ax4.set_ylim(0, 100)
    ax4.axhline(70, color='red', linestyle='--', linewidth=1)
    ax4.axhline(30, color='green', linestyle='--', linewidth=1)
    ax4.set_title('🔥 Heatmap RSI (Overbought/Oversold)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('RSI')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()
