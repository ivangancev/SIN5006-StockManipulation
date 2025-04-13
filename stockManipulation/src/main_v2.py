import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

# ------------------------------------------------------------------------------
# 1. Carregamento e Reconstrução do Livro de Ofertas (Level 1)
# ------------------------------------------------------------------------------

def reconstruir_livro_ordens_level1(arquivo_orderbook, arquivo_message):
    """
    Reconstrói o Level 1 do livro de ofertas a partir dos arquivos LOBSTER.
    Retorna um DataFrame com timestamp, best_bid, best_bid_size, best_ask, best_ask_size.
    """
    orderbook_df = pd.read_csv(arquivo_orderbook, header=None)
    message_df = pd.read_csv(arquivo_message, header=None)
    message_df.columns = ['time', 'type', 'order_id', 'size', 'price', 'direction']#, 'nan'] # Ignorando a última coluna 'nan'

    historico_livro_ordens = []
    livro_ordens_atual = {'buy': {}, 'sell': {}}

    for index, row in message_df.iterrows():
        timestamp = row['time']
        tipo = row['type']
        tamanho = row['size']
        preco = row['price']
        direcao = row['direction']

        if tipo == 1:  # New limit order
            if direcao == 1:  # Buy
                livro_ordens_atual['buy'][preco] = livro_ordens_atual['buy'].get(preco, 0) + tamanho
            elif direcao == -1:  # Sell
                livro_ordens_atual['sell'][preco] = livro_ordens_atual['sell'].get(preco, 0) + tamanho
        elif tipo == 2:  # Cancel limit order
            if direcao == 1 and preco in livro_ordens_atual['buy']:
                livro_ordens_atual['buy'][preco] -= tamanho
                if livro_ordens_atual['buy'][preco] == 0:
                    del livro_ordens_atual['buy'][preco]
            elif direcao == -1 and preco in livro_ordens_atual['sell']:
                livro_ordens_atual['sell'][preco] -= tamanho
                if livro_ordens_atual['sell'][preco] == 0:
                    del livro_ordens_atual['sell'][preco]
        elif tipo == 3 or tipo == 4:  # Execute trade (aggressive or visible)
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
        elif tipo == 5:  # Cancel visible limit order (similar to type 2)
            if direcao == 1 and preco in livro_ordens_atual['buy']:
                livro_ordens_atual['buy'][preco] -= tamanho
                if livro_ordens_atual['buy'][preco] == 0:
                    del livro_ordens_atual['buy'][preco]
            elif direcao == -1 and preco in livro_ordens_atual['sell']:
                livro_ordens_atual['sell'][preco] -= tamanho
                if livro_ordens_atual['sell'][preco] == 0:
                    del livro_ordens_atual['sell'][preco]

        melhor_compra = max(livro_ordens_atual['buy']) if livro_ordens_atual['buy'] else np.nan
        tamanho_melhor_compra = livro_ordens_atual['buy'].get(melhor_compra, 0) if livro_ordens_atual['buy'] else np.nan
        melhor_venda = min(livro_ordens_atual['sell']) if livro_ordens_atual['sell'] else np.nan
        tamanho_melhor_venda = livro_ordens_atual['sell'].get(melhor_venda, 0) if livro_ordens_atual['sell'] else np.nan

        historico_livro_ordens.append({
            'timestamp': timestamp,
            'best_bid': melhor_compra,
            'best_bid_size': tamanho_melhor_compra,
            'best_ask': melhor_venda,
            'best_ask_size': tamanho_melhor_venda
        })

    return pd.DataFrame(historico_livro_ordens).dropna(subset=['best_bid', 'best_ask']).reset_index(drop=True)

# ------------------------------------------------------------------------------
# 2. Geração de Janelas Deslizantes
# ------------------------------------------------------------------------------

def criar_janelas_deslizantes(dados, tamanho_janela, passo):
    """
    Cria janelas deslizantes a partir dos dados.
    Args:
        dados (pd.DataFrame): DataFrame contendo a série temporal.
        tamanho_janela (int): Tamanho de cada janela.
        passo (int): Passo entre as janelas.
    Returns:
        list: Lista de arrays NumPy representando as janelas.
    """
    janelas = []
    num_pontos = len(dados)
    for i in range(0, num_pontos - tamanho_janela + 1, passo):
        janela = dados[i:i + tamanho_janela].to_numpy()
        janelas.append(janela)
    return janelas

# ------------------------------------------------------------------------------
# 3. Geração de Recurrence Plots
# ------------------------------------------------------------------------------

def gerar_recurrence_plots(janelas, time_delay=1, dimension=1, threshold='point', percentage=10):
    """
    Gera Recurrence Plots para uma lista de janelas deslizantes.
    Args:
        janelas (list): Lista de arrays NumPy representando as janelas.
        time_delay (int): Atraso para o embedding.
        dimension (int): Dimensão do embedding.
        threshold (str or float): Limiar para recorrência.
        percentage (float): Percentual para o limiar pointwise.
    Returns:
        list: Lista de arrays NumPy representando os Recurrence Plots.
    """
    recurrence_plots = []
    rp = RecurrencePlot(time_delay=time_delay, dimension=dimension, threshold=threshold, percentage=percentage)
    for janela in janelas:
        # RecurrencePlot espera uma entrada 2D: [n_samples, n_timestamps]
        # Se a janela tiver múltiplas features, precisamos decidir como processar
        # Aqui, vamos gerar um RP para cada feature separadamente e armazenar
        rps_janela = []
        if janela.ndim > 1:
            for feature in range(janela.shape[1]):
                rp_feature = rp.fit_transform(janela[:, feature].reshape(1, -1))[0]
                rps_janela.append(rp_feature)
            recurrence_plots.append(np.mean(rps_janela, axis=0) if rps_janela else None) # Média dos RPs das features
        elif janela.ndim == 1:
            recurrence_plot = rp.fit_transform(janela.reshape(1, -1))[0]
            recurrence_plots.append(recurrence_plot)
        else:
            recurrence_plots.append(None)
    return [rp for rp in recurrence_plots if rp is not None]

# ------------------------------------------------------------------------------
# 4. Impressão dos Gráficos de Recurrence Plots
# ------------------------------------------------------------------------------

def visualizar_recurrence_plots(recurrence_plots, num_graficos=5):
    """
    Visualiza os Recurrence Plots gerados.
    Args:
        recurrence_plots (list): Lista de arrays NumPy representando os RPs.
        num_graficos (int): Número de gráficos para exibir.
    """
    num_plots = min(num_graficos, len(recurrence_plots))
    plt.figure(figsize=(15, 5 * num_plots))
    for i in range(num_plots):
        plt.subplot(num_plots, 1, i + 1)
        plt.imshow(recurrence_plots[i], cmap='binary', origin='lower')
        plt.title(f'Recurrence Plot da Janela {i + 1}')
        plt.xlabel('Tempo')
        plt.ylabel('Tempo')
        plt.colorbar(label='Recorrência')
    #plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# Função Principal (main)
# ------------------------------------------------------------------------------

def main():
    """
    Função principal para carregar dados, reconstruir o livro de ofertas,
    gerar janelas deslizantes, calcular recurrence plots e visualizá-los.
    """
    try:
        """ DADOS ORIGINAIS 
        arquivo_orderbook = "D:\Temp\LOBSTER_SampleFile_AMZN_2012-06-21_1\AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_message = "D:\Temp\LOBSTER_SampleFile_AMZN_2012-06-21_1\AMZN_2012-06-21_34200000_57600000_message_1.csv"
        tamanho_janela = 100
        passo = 50
        time_delay = 1
        dimension = 1
        threshold = 'point'
        percentage = 10
        num_graficos_para_visualizar = 5
        """
        
        """DADOS PARA TESTES"""
        #arquivo_message = "D:\Temp\LOBSTER_SampleFile_AMZN_2012-06-21_1\AMZN_2012-06-21_34200000_57600000_message_1.csv"
        arquivo_orderbook = "D:\Temp\LOBSTER_SampleFile_AMZN_2012-06-21_1\AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
        tamanho_janela = 500
        passo = 300
        time_delay = 1
        dimension = 1
        threshold = 'point'
        percentage = 65
        num_graficos_para_visualizar = 5

        #print("Carregando e reconstruindo o livro de ofertas...")
        #df_livro_ordens = reconstruir_livro_ordens_level1(arquivo_orderbook, arquivo_message)
        #print("Livro de ofertas reconstruído com sucesso.")

        # Abordagem com livro de ofertas simplificado
        print("Carregando o livro de ofertas...")
        df_livro_ordens = pd.read_csv(arquivo_orderbook, header=None)
        df_livro_ordens.columns = ['best_ask', 'best_ask_size','best_bid', 'best_bid_size' ]
        df_livro_ordens['best_bid'] = df_livro_ordens['best_bid'].astype(float)
        df_livro_ordens['best_bid_size'] = df_livro_ordens['best_bid_size'].astype(float)
        print("Livro de ofertas carregado com sucesso.")
        
        print(df_livro_ordens.head())

        print("\nCriando janelas deslizantes...")
        colunas_para_janelas = ['best_bid', 'best_ask']
        dados_para_janelas = df_livro_ordens[colunas_para_janelas]
        janelas_deslizantes = criar_janelas_deslizantes(dados_para_janelas, tamanho_janela, passo)
        print(f"{len(janelas_deslizantes)} janelas deslizantes criadas.")

        print("\nGerando recurrence plots...")
        recurrence_plots = gerar_recurrence_plots(janelas_deslizantes, time_delay, dimension, threshold, percentage)
        print(f"{len(recurrence_plots)} recurrence plots gerados.")

        if recurrence_plots:
            print("\nVisualizando os recurrence plots...")
            visualizar_recurrence_plots(recurrence_plots, num_graficos_para_visualizar)
        else:
            print("\nNenhum recurrence plot foi gerado.")

    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()