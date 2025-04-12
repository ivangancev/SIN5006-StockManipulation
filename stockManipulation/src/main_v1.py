import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

def carregar_dados_lobster(arquivo_orderbook, arquivo_message):
    # Carregue os dados dos arquivos orderbook e message
    orderbook_df = pd.read_csv(arquivo_orderbook, header=None)
    message_df = pd.read_csv(arquivo_message, header=None)

def reconstruir_e_preprocessar_livro_ordens(arquivo_orderbook, arquivo_message, niveis_profundidade=1):
    """
    Reconstrói o livro de ordens (até um certo nível de profundidade)
    e pré-processa os dados para análise.

    Args:
        arquivo_orderbook (str): Caminho para o arquivo orderbook.
        arquivo_message (str): Caminho para o arquivo message.
        niveis_profundidade (int): Número de níveis de profundidade do livro de ordens
                                   a serem reconstruídos (melhores ofertas de compra e venda).

    Returns:
        pd.DataFrame: DataFrame com o histórico do livro de ordens reconstruído e pré-processado.
    """

    orderbook_df = pd.read_csv(arquivo_orderbook, header=None)
    message_df = pd.read_csv(arquivo_message, header=None)
#    message_df.columns = ['time', 'type', 'order_id', 'size', 'price', 'direction', 'nan']
    message_df.columns = ['time', 'type', 'order_id', 'size', 'price', 'direction']

    historico_livro_ordens = []
    livro_ordens_atual = {'buy': {}, 'sell': {}}

    for index, row in message_df.iterrows():
        timestamp = row['time']
        tipo = row['type']
        ordem_id = row['order_id']
        tamanho = row['size']
        preco = row['price']
        direcao = row['direction']

        if tipo == 1:  # New limit order
            if direcao == 1:  # Buy
                if preco not in livro_ordens_atual['buy']:
                    livro_ordens_atual['buy'][preco] = 0
                livro_ordens_atual['buy'][preco] += tamanho
            elif direcao == -1:  # Sell
                if preco not in livro_ordens_atual['sell']:
                    livro_ordens_atual['sell'][preco] = 0
                livro_ordens_atual['sell'][preco] += tamanho
        elif tipo == 2:  # Cancel limit order
            if direcao == 1 and preco in livro_ordens_atual['buy']:
                livro_ordens_atual['buy'][preco] -= tamanho
                if livro_ordens_atual['buy'][preco] == 0:
                    del livro_ordens_atual['buy'][preco]
            elif direcao == -1 and preco in livro_ordens_atual['sell']:
                livro_ordens_atual['sell'][preco] -= tamanho
                if livro_ordens_atual['sell'][preco] == 0:
                    del livro_ordens_atual['sell'][preco]
        elif tipo == 3:  # Execute trade (aggressive order)
            if direcao == 1 and livro_ordens_atual['sell']:
                melhor_venda = min(livro_ordens_atual['sell'])
                livro_ordens_atual['sell'][melhor_venda] -= tamanho
                if livro_ordens_atual['sell'][melhor_venda] == 0:
                    del livro_ordens_atual['sell'][melhor_venda]
            elif direcao == -1 and livro_ordens_atual['buy']:
                melhor_compra = max(livro_ordens_atual['buy'])
                livro_ordens_atual['buy'][melhor_compra] -= tamanho
                if livro_ordens_atual['buy'][melhor_compra] == 0:
                    del livro_ordens_atual['buy'][melhor_compra]
        elif tipo == 4:  # Execute visible limit order
            if direcao == 1 and preco in livro_ordens_atual['sell']:
                livro_ordens_atual['sell'][preco] -= tamanho
                if livro_ordens_atual['sell'][preco] == 0:
                    del livro_ordens_atual['sell'][preco]
            elif direcao == -1 and preco in livro_ordens_atual['buy']:
                livro_ordens_atual['buy'][preco] -= tamanho
                if livro_ordens_atual['buy'][preco] == 0:
                    del livro_ordens_atual['buy'][preco]
        elif tipo == 5:  # Cancel visible limit order
            if direcao == 1 and preco in livro_ordens_atual['buy']:
                livro_ordens_atual['buy'][preco] -= tamanho
                if livro_ordens_atual['buy'][preco] == 0:
                    del livro_ordens_atual['buy'][preco]
            elif direcao == -1 and preco in livro_ordens_atual['sell']:
                livro_ordens_atual['sell'][preco] -= tamanho
                if livro_ordens_atual['sell'][preco] == 0:
                    del livro_ordens_atual['sell'][preco]
        elif tipo == 7:  # Cross trade (usually internal)
            if direcao == 1 and livro_ordens_atual['sell']:
                melhor_venda = min(livro_ordens_atual['sell'])
                if preco == melhor_venda:
                    livro_ordens_atual['sell'][melhor_venda] -= tamanho
                    if livro_ordens_atual['sell'][melhor_venda] == 0:
                        del livro_ordens_atual['sell'][melhor_venda]
            elif direcao == -1 and livro_ordens_atual['buy']:
                melhor_compra = max(livro_ordens_atual['buy'])
                if preco == melhor_compra:
                    livro_ordens_atual['buy'][melhor_compra] -= tamanho
                    if livro_ordens_atual['buy'][melhor_compra] == 0:
                        del livro_ordens_atual['buy'][melhor_compra]

        # Registrar o estado do livro de ordens (até o nível de profundidade desejado)
        snapshot = {'time': timestamp}
        melhores_compras = sorted(livro_ordens_atual['buy'].items(), key=lambda item: item[0], reverse=True)[:niveis_profundidade]
        melhores_vendas = sorted(livro_ordens_atual['sell'].items(), key=lambda item: item[0])[:niveis_profundidade]

        for i in range(niveis_profundidade):
            if i < len(melhores_compras):
                snapshot[f'bid_price_{i+1}'] = melhores_compras[i][0]
                snapshot[f'bid_size_{i+1}'] = melhores_compras[i][1]
            else:
                snapshot[f'bid_price_{i+1}'] = np.nan
                snapshot[f'bid_size_{i+1}'] = np.nan

            if i < len(melhores_vendas):
                snapshot[f'ask_price_{i+1}'] = melhores_vendas[i][0]
                snapshot[f'ask_size_{i+1}'] = melhores_vendas[i][1]
            else:
                snapshot[f'ask_price_{i+1}'] = np.nan
                snapshot[f'ask_size_{i+1}'] = np.nan

        historico_livro_ordens.append(snapshot)

    df_historico = pd.DataFrame(historico_livro_ordens)

    # Pré-processamento adicional (opcional):
    # - Calcular mid-price, spread, etc.
    df_historico['mid_price'] = (df_historico['bid_price_1'] + df_historico['ask_price_1']) / 2
    df_historico['spread'] = df_historico['ask_price_1'] - df_historico['bid_price_1']

    # - Remover linhas com NaN iniciais (antes da primeira ordem ser registrada)
    df_historico.dropna(subset=['bid_price_1', 'ask_price_1'], inplace=True)

    return df_historico


#    dados_processados = {
#        "orderbook": orderbook_df,
#        "message": message_df
#    }

#    return dados_processados

def criar_janelas_deslizantes(dados, tamanho_janela, passo):
    janelas = []
    for i in range(0, len(dados) - tamanho_janela + 1, passo):
        janela = dados[i:i + tamanho_janela]
        janelas.append(janela)
    #num_janelas_para_inspecionar = 5
    #for i in range(min(num_janelas_para_inspecionar, len(janelas))):
    #    print(f"Janela {i+1}:")
    #    print(janelas[i])
    #    plt.figure()
    #    plt.plot(janelas[i])
    #    plt.title(f"Série Temporal da Janela {i+1}")
    #    plt.xlabel("Tempo (Pontos de Dados na Janela)")
    #    plt.ylabel("Valor")
    #    plt.show()
    
    return janelas

def gerar_recurrence_plots(janelas, time_delay=1, dimension=1):
    recurrence_plots = []
#    rp = RecurrencePlot(time_delay=1, dimension=1)  # Ajuste os parâmetros conforme necessário
    for janela in janelas:
        rp = RecurrencePlot(time_delay=time_delay, dimension=dimension)  # Ajuste os parâmetros conforme necessário
        recurrence_plot = rp.fit_transform(janela)
        recurrence_plots.append(recurrence_plot[0])  # RecurrencePlot.fit_transform retorna uma lista de arrays
    return recurrence_plots

def visualizar_recurrence_plots(recurrence_plots, num_graficos=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_graficos):
        plt.plot(recurrence_plots[i])
        plt.subplot(1, num_graficos, i + 1)
        plt.imshow(recurrence_plots[i], cmap='binary', origin='lower')
        plt.title(f'Recurrence Plot {i + 1}')
    plt.show()

def main():
    print("Welcome to my Python project!")

    # Carregar dados do LOBSTER
    try:
        
        #Carga do arquivo e reconstrução do livro de ordens
        arquivo_orderbook = "D:\Temp\LOBSTER_SampleFile_AMZN_2012-06-21_1\AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
        arquivo_message = "D:\Temp\LOBSTER_SampleFile_AMZN_2012-06-21_1\AMZN_2012-06-21_34200000_57600000_message_1.csv"
        #dados_lobster = carregar_dados_lobster(arquivo_orderbook, arquivo_message)
        #print("Dados carregados com sucesso!")
        # Exemplo de uso:
        #arquivo_orderbook = 'your_orderbook_file.csv'  # Substitua pelo seu arquivo
        #arquivo_message = 'your_message_file.csv'      # Substitua pelo seu arquivo
        historico_reconstruido = reconstruir_e_preprocessar_livro_ordens(arquivo_orderbook, arquivo_message, niveis_profundidade=1)
        print(historico_reconstruido.head())

        #Cria janelas deslizantes
        tamanho_janela = 4  # Ajuste o tamanho da janela conforme necessário
        passo = 2  # Ajuste o passo conforme necessário
        janelas_deslizantes = criar_janelas_deslizantes(historico_reconstruido, tamanho_janela, passo)

        #cria os gráficos de recorrência
        recurrence_plots = gerar_recurrence_plots(janelas_deslizantes)

        # Visualiza os gráficos de recorrência
        visualizar_recurrence_plots(recurrence_plots)

    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()