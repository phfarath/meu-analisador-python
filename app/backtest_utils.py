def backtest_avancado(df_test, stop_loss=0.01, take_profit=0.02, trailing_stop=True):
    capital = 10000
    retornos = []
    entradas = []
    saidas = []
    tipos_saida = []
    posicao_aberta = False

    for i in range(len(df_test) - 5):
        if posicao_aberta:
            continue

        if (df_test['sinal_compra'].iloc[i] == 1 and
            df_test['filtros_ok'].iloc[i] and
            df_test['proba_alta'].iloc[i] > 0.6):

            preco_entrada = df_test['close'].iloc[i]
            preco_alvo = preco_entrada * (1 + take_profit)
            preco_stop = preco_entrada * (1 - stop_loss)
            stop_atual = preco_stop

            for j in range(1, min(10, len(df_test) - i)):
                preco_atual = df_test['close'].iloc[i + j]

                if trailing_stop and preco_atual > preco_entrada:
                    novo_stop = preco_entrada + (preco_atual - preco_entrada) * 0.5
                    if novo_stop > stop_atual:
                        stop_atual = novo_stop

                if preco_atual >= preco_alvo:
                    saida = preco_atual
                    tipo_saida = 'TP'
                    posicao_aberta = False
                    break
                elif preco_atual <= stop_atual:
                    saida = preco_atual
                    tipo_saida = 'SL' if stop_atual == preco_stop else 'TS'
                    posicao_aberta = False
                    break

                if j == min(10, len(df_test) - i) - 1:
                    saida = preco_atual
                    tipo_saida = 'Tempo'
                    posicao_aberta = False

            retorno = (saida - preco_entrada) / preco_entrada
            entradas.append((df_test.index[i], df_test['low'].iloc[i] * 0.995))
            saidas.append((df_test.index[i + j], df_test['high'].iloc[i + j] * 1.002))
            tipos_saida.append(tipo_saida)
            retornos.append(retorno)
        else:
            retornos.append(0)

    # Padroniza tamanho das listas
    min_len = min(len(entradas), len(saidas), len(tipos_saida))
    retornos = retornos[:min_len]
    entradas = entradas[:min_len]
    saidas = saidas[:min_len]
    tipos_saida = tipos_saida[:min_len]

    return retornos, entradas, saidas, tipos_saida

