"""
Classe principal do robô de trading.
Implementa a lógica de negociação e gerenciamento de operações.
"""

import ta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from logger import logger
from aplicar_filtros import aplicar_filtros_tecnicos
from backtest_agressivo import backtest_agressivo
from backtest_utils import backtest_avancado
from analisar_desempenho import calculo_desempenho, analisar_drawdown, analisar_risco

class RoboTrading:
    """
    Classe principal do robô de trading.
    
    Esta classe implementa todas as funcionalidades necessárias para:
    - Preparação e processamento de dados
    - Treinamento do modelo de machine learning
    - Execução de backtest
    - Gerenciamento de operações
    
    Attributes:
        capital_inicial (float): Capital inicial para operações
        risco_por_trade (float): Risco por trade
        stop_loss (float): Stop loss
        take_profit (float): Take profit
        trailing_stop (bool): Trailing stop
        modelo (RandomForestClassifier): Modelo de machine learning
        scaler (StandardScaler): Normalizador de dados
        historico_trades (list): Lista de trades realizados
    """
    
    def __init__(self, capital_inicial=10000, risco_por_trade=0.02,
                 stop_loss=0.01, take_profit=0.02, trailing_stop=True):
        """
        Inicializa o robô de trading com os parâmetros básicos.
        
        Args:
            capital_inicial (float): Capital inicial para operações
            risco_por_trade (float): Risco por trade
            stop_loss (float): Stop loss
            take_profit (float): Take profit
            trailing_stop (bool): Trailing stop
        """
        self.capital = capital_inicial
        self.risco_por_trade = risco_por_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.modelo = None
        self.scaler = StandardScaler()
        self.historico_trades = []
        
        logger.info(f"Robô inicializado com capital: R${capital_inicial:.2f}")
        logger.info(f"Risco por trade: {risco_por_trade*100:.1f}%")
        logger.info(f"Stop Loss: {stop_loss*100:.1f}%")
        logger.info(f"Take Profit: {take_profit*100:.1f}%")
        logger.info(f"Trailing stop: {'Sim' if trailing_stop else 'Não'}")

    def preparar_dados(self, df):
        """
        Prepara os dados para treinamento do modelo.
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            
        Returns:
            pd.DataFrame: DataFrame processado e pronto para treinamento
        """
        logger.info("Preparando dados para treinamento...")
        df = self.adicionar_indicadores(df)
        df['target'] = df['close'].shift(-3)
        df['target_class'] = np.where(df['target'] > df['close'] * 1.002, 1,
                                     np.where(df['target'] < df['close'] * 0.998, 0, -1))
        df = df[df['target_class'] != -1]
        df = aplicar_filtros_tecnicos(df)
        
        logger.info(f"Dados preparados. Shape final: {df.shape}")
        df = aplicar_filtros_tecnicos(df)
        df['target_class'] = (df['close'].shift(-1) > df['close']).astype(int)
        return df

    def adicionar_indicadores(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['volume_change'] = df['volume'].pct_change()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        return df

    def treinar_modelo(self, df):
        """
        Treina o modelo de machine learning.
        
        Args:
            df (pd.DataFrame): DataFrame com dados preparados
            
        Returns:
            np.array: Probabilidades de previsão
        """
        logger.info("Iniciando treinamento do modelo...")
        
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal',
                   'sma_20', 'ema_20', 'bb_upper', 'bb_lower', 'atr', 'volume_change',
                   'obv', 'adx']
        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df['target_class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        X_train_scaled = self.scaler.fit_transform(X_train)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_grid={
                'n_estimators': [100, 200],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_res, y_res)
        self.modelo = grid_search.best_estimator_
        X_test_scaled = self.scaler.transform(X_test)
        probs = self.modelo.predict_proba(X_test_scaled)[:, 1]
        
        logger.info("Modelo treinado com sucesso!")
        return probs

    def executar_backtest(self, df, probs, modo="padrao"):
        """
        Executa backtest com os dados e previsões.
        
        Args:
            df (pd.DataFrame): DataFrame com dados históricos
            probs (np.array): Probabilidades de previsão
            modo (str): Modo de operação ('super_agressivo' ou 'conservador')
            
        Returns:
            tuple: (métricas, DataFrame de teste, entradas, saídas)
        """
        logger.info(f"Iniciando backtest no modo {modo}...")
        
        df_test = df.iloc[-len(probs):].copy()
        df_test['proba_alta'] = probs
        df_test['sinal_compra'] = (df_test['proba_alta'] > 0.6).astype(int)

        if modo == "super_agressivo":
            df_test['filtros_ok'] = True

        print("📊 Filtros OK por linha:", df_test['filtros_ok'].value_counts())
        print("✅ Sinais de compra:", df_test['sinal_compra'].sum())
        print("✅ Combinação válida (compra + filtro):", ((df_test['sinal_compra'] == 1) & (df_test['filtros_ok'])).sum())

        if modo in ["agressivo", "super_agressivo"]:
            retornos, entradas, saidas, tipos_saida = backtest_agressivo(
                df_test,
                probs,
                capital_inicial=self.capital,
                stop_loss_pct=self.stop_loss,
                take_profit_pct=self.take_profit
            )
        else:
            retornos, entradas, saidas, tipos_saida = backtest_avancado(
                df_test,
                probs,
                capital_inicial=self.capital,
                stop_loss_pct=self.stop_loss,
                take_profit_pct=self.take_profit
            )

        if not (len(retornos) == len(entradas) == len(saidas) == len(tipos_saida)):
            print("⚠️ Inconsistência nos resultados do backtest. Ignorando histórico.")
        else:
            for i, retorno in enumerate(retornos):
                if retorno != 0:
                    self.historico_trades.append({
                        'data_entrada': entradas[i][0],
                        'data_saida': saidas[i][0],
                        'tipo_saida': tipos_saida[i],
                        'retorno': retorno
                    })

        if len(retornos) == len(tipos_saida):
            print(f"📦 Total retornos: {len(retornos)}")
            print(f"📦 Total tipos_saida: {len(tipos_saida)}")
            print(f"📦 Trades positivos: {sum([1 for r in retornos if r != 0])}")
            print(f"📦 Tipos únicos: {set(tipos_saida)}")
            print(f"✅ Verificação manual dos retornos > 0: {[r for r in retornos if r > 0]}")

            metricas = calculo_desempenho(retornos, tipos_saida)
        else:
            print("⚠️ Não foi possível calcular métricas — listas inconsistentes.")
            metricas = {'total_trades': 0}

        logger.info("Backtest concluído!")
        return metricas, df_test, entradas, saidas

    def monitorar_mercado(self, dados_atuais):
        """
        Monitora o mercado em tempo real e gera sinais.
        
        Args:
            dados_atuais (pd.DataFrame): Dados atuais do mercado
            
        Returns:
            dict or None: Sinal de operação ou None se não houver sinal
        """
        logger.info("Monitorando mercado...")
        
        # Implementar lógica de monitoramento aqui
        # ...
        
        return sinal