"""
Classe para gerenciar a conexão e operações com a corretora.
Implementa todas as funcionalidades necessárias para interagir com a API da corretora.
"""

import requests
import time
import hmac
import hashlib
from datetime import datetime
from logger import logger

class ConexaoCorretora:
    """
    Classe para gerenciar a conexão com a corretora.
    
    Esta classe implementa todas as funcionalidades necessárias para:
    - Autenticação com a API
    - Obtenção de dados de mercado
    - Envio de ordens
    - Consulta de posições e saldo
    
    Attributes:
        api_key (str): Chave da API
        api_secret (str): Segredo da API
        base_url (str): URL base da API
        session (requests.Session): Sessão HTTP
    """
    
    def __init__(self, api_key, api_secret, base_url='https://api.exchange.com'):
        """
        Inicializa a conexão com a corretora.
        
        Args:
            api_key (str): Chave da API
            api_secret (str): Segredo da API
            base_url (str): URL base da API
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        
        logger.info("Conexão com corretora inicializada")
    
    def _gerar_assinatura(self, params):
        """
        Gera assinatura HMAC-SHA256 para autenticação.
        
        Args:
            params (dict): Parâmetros da requisição
            
        Returns:
            str: Assinatura gerada
        """
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def obter_dados_mercado(self, simbolo, timeframe, limite=1000):
        """
        Obtém dados históricos do mercado.
        
        Args:
            simbolo (str): Símbolo do ativo (ex: 'BTC-USD')
            timeframe (str): Intervalo de tempo (ex: '15min')
            limite (int): Número máximo de candles
            
        Returns:
            pd.DataFrame: DataFrame com os dados históricos
        """
        logger.info(f"Obtendo dados de {simbolo} no timeframe {timeframe}")
        
        endpoint = f"{self.base_url}/market/history"
        params = {
            'symbol': simbolo,
            'timeframe': timeframe,
            'limit': limite,
            'timestamp': int(time.time() * 1000)
        }
        
        params['signature'] = self._gerar_assinatura(params)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            dados = response.json()
            
            logger.info(f"Dados obtidos com sucesso: {len(dados)} candles")
            return pd.DataFrame(dados)
            
        except Exception as e:
            logger.error(f"Erro ao obter dados: {str(e)}")
            raise
    
    def enviar_ordem(self, simbolo, tipo, quantidade, preco=None, 
                    stop_loss=None, take_profit=None):
        """
        Envia uma ordem para a corretora.
        
        Args:
            simbolo (str): Símbolo do ativo
            tipo (str): Tipo de ordem ('compra' ou 'venda')
            quantidade (float): Quantidade a ser negociada
            preco (float, optional): Preço limite
            stop_loss (float, optional): Preço do stop loss
            take_profit (float, optional): Preço do take profit
            
        Returns:
            dict: Resposta da corretora
        """
        logger.info(f"Enviando ordem de {tipo} para {simbolo}")
        
        endpoint = f"{self.base_url}/order"
        params = {
            'symbol': simbolo,
            'side': tipo.upper(),
            'quantity': quantidade,
            'timestamp': int(time.time() * 1000)
        }
        
        if preco:
            params['price'] = preco
            params['type'] = 'LIMIT'
        else:
            params['type'] = 'MARKET'
            
        if stop_loss:
            params['stopLoss'] = stop_loss
        if take_profit:
            params['takeProfit'] = take_profit
            
        params['signature'] = self._gerar_assinatura(params)
        
        try:
            response = self.session.post(endpoint, json=params)
            response.raise_for_status()
            resultado = response.json()
            
            logger.info(f"Ordem enviada com sucesso: {resultado['orderId']}")
            return resultado
            
        except Exception as e:
            logger.error(f"Erro ao enviar ordem: {str(e)}")
            raise
    
    def verificar_posicoes(self):
        """
        Verifica as posições abertas.
        
        Returns:
            list: Lista de posições abertas
        """
        logger.info("Verificando posições abertas...")
        
        endpoint = f"{self.base_url}/position"
        params = {
            'timestamp': int(time.time() * 1000)
        }
        
        params['signature'] = self._gerar_assinatura(params)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            posicoes = response.json()
            
            logger.info(f"Posições encontradas: {len(posicoes)}")
            return posicoes
            
        except Exception as e:
            logger.error(f"Erro ao verificar posições: {str(e)}")
            raise
    
    def obter_saldo(self):
        """
        Obtém o saldo da conta.
        
        Returns:
            dict: Saldo da conta
        """
        logger.info("Obtendo saldo da conta...")
        
        endpoint = f"{self.base_url}/account/balance"
        params = {
            'timestamp': int(time.time() * 1000)
        }
        
        params['signature'] = self._gerar_assinatura(params)
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            saldo = response.json()
            
            logger.info(f"Saldo obtido com sucesso")
            return saldo
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {str(e)}")
            raise