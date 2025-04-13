import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

# ------------------------------------------------------------------------------
# 1. Carga do Livro de Ofertas (Level 1)
# ------------------------------------------------------------------------------

def carregar_livro_ordens(arquivo_orderbook):
    """
    Carrega o livro de ordens a partir de um arquivo CSV de acordo com o layout LOBSTER.
    Args:
        arquivo_orderbook (str): Caminho para o arquivo CSV do livro de ordens.
    Returns:
        pd.DataFrame: DataFrame contendo o livro de ordens.
    """

    livro_ordens = pd.read_csv(arquivo_orderbook, header=None)
    livro_ordens.columns = ['best_ask', 'best_ask_size','best_bid', 'best_bid_size' ]
    livro_ordens['best_bid'] = livro_ordens['best_bid'].astype(float)
    livro_ordens['best_bid_size'] = livro_ordens['best_bid_size'].astype(float)

    return livro_ordens

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

def gerar_recurrence_plots(janelas, time_delay, dimension, threshold, percentage):
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
        plt.subplot(1, num_plots, i + 1)
        plt.imshow(recurrence_plots[i], cmap='binary', origin='lower')
        plt.title(f'Recurrence Plot da Janela {i + 1}')
        plt.xlabel('Tempo')
        plt.ylabel('Tempo')
        plt.colorbar(label='Recorrência', shrink=0.3)
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------------------------
# Função Principal (main)
# ------------------------------------------------------------------------------

def main():
    """
    Função principal para carregar dados, gerar janelas deslizantes, 
    calcular recurrence plots e visualizá-los.
    """
    try:
        """ PARÂMETROS ORIGINAIS 
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
        
        """PARAMÊTROS PARA EXECUÇÃO"""
        #arquivo_orderbook = "D:\Temp\Lobster data\AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\AaPL_2012-06-21_34200000_57600000_orderbook_1.csv"
        arquivo_orderbook = "D:\Temp\Lobster data\GOOG_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\INTC_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\MSFT_2012-06-21_34200000_57600000_orderbook_1.csv"
        tamanho_janela = 800
        passo = 500
        time_delay = 1
        dimension = 1
        threshold = 'point'
        percentage = 50
        num_graficos_para_visualizar = 4

        # Abordagem com livro de ofertas simplificado
        print("Carregando o livro de ofertas...")
        df_livro_ordens = carregar_livro_ordens(arquivo_orderbook)
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