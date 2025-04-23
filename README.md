# Meu Analisador de Trading

Este é um sistema de análise e trading automatizado que utiliza machine learning para prever movimentos de mercado e executar operações.

## 📁 Estrutura do Projeto

```
meu_analisador/
├── app/                    # Código principal da aplicação
│   ├── main.py            # Script principal de execução
│   ├── classeRobo.py      # Classe principal do robô de trading
│   ├── processar_dados.py # Processamento de dados e indicadores
│   ├── aplicar_filtros.py # Filtros de mercado
│   ├── backtest_utils.py  # Utilitários para backtest
│   ├── backtest_agressivo.py # Estratégia de backtest agressiva
│   ├── analisar_desempenho.py # Análise de performance
│   ├── visualizar_trades.py # Visualização de trades
│   └── graficos.py        # Funções de plotagem
├── data/                  # Dados históricos e datasets
├── notebooks/            # Jupyter notebooks para análise
└── requirements.txt      # Dependências do projeto
```

## 🚀 Funcionalidades

- Coleta e processamento de dados de mercado
- Geração de indicadores técnicos
- Modelo de machine learning para previsão de movimentos
- Sistema de backtest com diferentes estratégias
- Visualização de trades e análise de desempenho

## 🛠️ Requisitos

- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

## 📦 Instalação

1. Clone o repositório
2. Crie um ambiente virtual:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Uso

Execute o script principal:
```bash
python app/main.py
```

## 📊 Notebooks

Os notebooks na pasta `notebooks/` contêm análises e experimentos:
- `backtest_com_sl_tp_setas.ipynb`: Backtest com stop loss e take profit
- `modelo_ativos_melhorado_completo.ipynb`: Modelo completo de análise
- `previsao_confiante_com_setas.ipynb`: Análise de previsões

## 🤝 Contribuição

Sinta-se à vontade para contribuir com melhorias e novas funcionalidades. 