"""
Arquivo de configuração central do projeto.
Aqui estão definidas todas as configurações globais e parâmetros do sistema.
"""

import logging

# Configuração de logging
LOG_CONFIG = {
    'level': 'INFO',
    'file': 'trading.log',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Configurações do YFinance
YFINANCE_CONFIG = {
    'ticker': 'BTC-USD',          # Alterado para Bitcoin
    'periodo': '90d',             # Aumentado para 90 dias para ter mais dados
    'intervalo': '15min',         # Mantido 15 minutos
    'proxy': None,
    'timeout': 30
}

# Configurações de Trading
TRADING_CONFIG = {
    'capital_inicial': 10000,
    'risco_por_trade': 0.02,      # 2% do capital por trade
    'stop_loss_pct': 0.02,        # Reduzido para 1.5% para sair mais rápido de operações ruins
    'take_profit_pct': 0.06,      # Reduzido para 4.5% para garantir mais vitórias
    'trailing_stop': True,
    'trailing_stop_offset': 0.01   # Aumentado para 1% para BTC
}

# Configurações do Modelo
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 300,           # Aumentado para 300 árvores
    'max_depth': 12,               # Reduzido para evitar overfitting
    'min_samples_split': 10,       # Aumentado para mais robustez
    'confidence_threshold': 0.65    # Aumentado para 65% de confiança
}

# Configurações de Backtest
BACKTEST_CONFIG = {
    'modo_padrao': 'agressivo',   # Mantido modo agressivo
    'comissao': 0.001,           # 0.1% de comissão
    'slippage': 0.001,           # Aumentado para 0.1% devido à menor liquidez
    'alavancagem': 2.0           # Mantida alavancagem de 2x
}

# Configurações de Visualização
VISUALIZATION_CONFIG = {
    'show_plots': True,
    'save_plots': True,
    'plot_format': 'png',
    'plot_width': 1200,
    'plot_height': 800
} 