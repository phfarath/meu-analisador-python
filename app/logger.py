"""
Módulo de logging para o sistema de trading.
Configura e fornece funções para registro de eventos e operações.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from config import LOG_CONFIG


def setup_logger(name='trading_bot'):
    """
    Configura e retorna um logger com as configurações especificadas.
    
    Args:
        name (str): Nome do logger
        
    Returns:
        logging.Logger: Logger configurado
    """
        # Criar diretório de logs se não existir
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configura o logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG['level']))
    
    # Formatar a data para o nome do arquivo
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'{date_str}_{LOG_CONFIG["file"]}')
    
        # Configurar handler para arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, LOG_CONFIG['level']))
    
    # Configurar handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_CONFIG['level']))
    
    # Configurar formato
    formatter = logging.Formatter(LOG_CONFIG['format'])
    file_handler.setFormatter(formatter)
    
    console_handler.setFormatter(formatter)
        
    # Adicionar handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Cria logger global
logger = setup_logger()

def log_trade(trade_type, price, quantity, stop_loss=None, take_profit=None):
    """
    Registra uma operação de trade.
    
    Args:
        trade_type (str): Tipo de operação (compra/venda)
        price (float): Preço da operação
        quantity (float): Quantidade negociada
        stop_loss (float, optional): Preço do stop loss
        take_profit (float, optional): Preço do take profit
    """
    logger.info(f"Operação de {trade_type} realizada:")
    logger.info(f"Preço: {price:.2f}")
    logger.info(f"Quantidade: {quantity:.4f}")
    if stop_loss:
        logger.info(f"Stop Loss: {stop_loss:.2f}")
    if take_profit:
        logger.info(f"Take Profit: {take_profit:.2f}")

def log_backtest_results(metricas):
    """
    Registra os resultados do backtest no log.
    
    Args:
        metricas (dict): Dicionário com as métricas do backtest
    """
    logger.info("Resultados do Backtest:")
    for metric, value in metricas.items():
        if isinstance(value, (int, float)):
            logger.info(f"{metric}: {value:.2f}")
        else:
            logger.info(f"{metric}: {value}")

def log_error(error_msg, exception=None):
    """
    Registra um erro ocorrido no sistema.
    
    Args:
        error_msg (str): Mensagem de erro
        exception (Exception, optional): Exceção ocorrida
    """
    logger.error(error_msg)
    if exception:
        logger.exception(exception) 