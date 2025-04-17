# Loop principal do robô de trading
def executar_robo():
    # Configurações
    api_key = 'SUA_API_KEY'
    api_secret = 'SUA_API_SECRET'
    ativo = 'AAPL'
    timeframe = '15min'
    capital_inicial = 10000

    # Inicializar componentes
    corretora = ConexaoCorretora(api_key, api_secret)
    robo = RoboTrading(capital_inicial=capital_inicial)

    # Carregar dados históricos para treinamento
    dados_historicos = corretora.obter_dados_mercado(ativo, timeframe, limite=5000)

    # Preparar e treinar modelo
    dados_processados = robo.preparar_dados(dados_historicos)
    probs = robo.treinar_modelo(dados_processados)

    # Executar backtest para validação
    metricas, _ = robo.executar_backtest(dados_processados, probs)
    print(f"Métricas do backtest: {metricas}")

    # Verificar se o modelo é bom o suficiente para operar
    if metricas['win_rate'] < 0.5 or metricas['profit_factor'] < 1.5:
        print("Modelo não atende aos critérios mínimos. Abortando operações.")
        return

    # Loop de monitoramento do mercado
    posicao_aberta = False
    while True:
        try:
            # Obter dados atualizados
            dados_atuais = corretora.obter_dados_mercado(ativo, timeframe, limite=100)

            # Verificar posições existentes
            posicoes = corretora.verificar_posicoes_abertas()
            posicao_aberta = any(p['ativo'] == ativo for p in posicoes)

            # Se não tiver posição aberta, procurar por novos sinais
            if not posicao_aberta:
                sinal = robo.monitorar_mercado(dados_atuais)

                if sinal:
                    print(f"Sinal gerado: {sinal}")
                    # Enviar ordem para a corretora
                    corretora.enviar_ordem(
                        ativo=sinal['ativo'],
                        tipo='compra',
                        quantidade=sinal['tamanho_posicao'] / sinal['preco_entrada'],
                        preco=sinal['preco_entrada'],
                        stop_loss=sinal['stop_loss'],
                        take_profit=sinal['take_profit']
                    )
                    print(f"Ordem enviada para {sinal['ativo']} a {sinal['preco_entrada']}")

            # Aguardar próxima verificação
            time.sleep(60)  # Verificar a cada minuto

        except Exception as e:
            print(f"Erro no loop principal: {e}")
            time.sleep(300)  # Esperar 5 minutos em caso de erro