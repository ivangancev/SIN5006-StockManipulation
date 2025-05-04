import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

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
    #livro_ordens['best_ask'] = livro_ordens['best_ask'].astype(float)
    #livro_ordens['best_bid'] = livro_ordens['best_bid'].astype(float)

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
    graph_index = 1
    for i in range(0, num_plots):
        plt.subplot(1, num_plots, i + 1)
        plt.imshow(recurrence_plots[graph_index], cmap='binary', origin='lower')
        #plt.imshow(recurrence_plots[i], cmap='binary', origin='lower')
        plt.title(f'Recurrence Plot da Janela {graph_index}')
        plt.xlabel('Tempo')
        plt.ylabel('Tempo')
        plt.colorbar(label='Recorrência', shrink=0.3)
        plt.tight_layout()
        graph_index = graph_index + int(round(len(recurrence_plots)/num_plots,0))
        plt.show()


# ------------------------------------------------------------------------------
# 5. Preparação dos Dados para o VAE
# ------------------------------------------------------------------------------

def preparar_dados_vae(recurrence_plots):
    """
    Prepara os recurrence plots para serem usados como entrada em um VAE.
    Args:
        recurrence_plots (list): Lista de arrays NumPy representando os RPs (T x T).
    Returns:
        tuple: Uma tupla contendo os dados de treinamento e teste (expandidos para CNN).
    """
    T = recurrence_plots[0].shape[0]
    input_shape = (T, T, 1)  # Adiciona uma dimensão de canal (grayscale)
    rp_array = np.array(recurrence_plots).astype('float32') / 1.0  # Normalizar para [0, 1]
    X_train, X_test = train_test_split(rp_array, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return X_train, X_test, input_shape

# ------------------------------------------------------------------------------
# 6. Construção do Variational Autoencoder (VAE)
# ------------------------------------------------------------------------------

def construir_vae(input_shape, latent_dim):
    """
    Constrói o modelo Variational Autoencoder (VAE).
    Args:
        input_shape (tuple): Forma da entrada (altura, largura, canais).
        latent_dim (int): Dimensão do espaço latente.
    Returns:
        tuple: Uma tupla contendo os modelos encoder, decoder e vae.
    """
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(encoder_inputs)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.keras.backend.shape(z_mean)[0]
        dim = tf.keras.backend.int_shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(np.prod(encoder.layers[-2].output_shape[1:]), activation='relu')(latent_inputs)
    x = keras.layers.Reshape(encoder.layers[-2].output_shape[1:])(x)
    x = keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    decoder_outputs = keras.layers.Conv2DTranspose(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

    # VAE modelo completo
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name='vae')

    # Função de perda do VAE
    reconstruction_loss = keras.losses.binary_crossentropy(keras.layers.Flatten()(encoder_inputs), keras.layers.Flatten()(outputs))
    reconstruction_loss *= input_shape[0] * input_shape[1]
    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    vae.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')

    return encoder, decoder, vae

# ------------------------------------------------------------------------------
# 7. Treinamento do VAE
# ------------------------------------------------------------------------------

def treinar_vae(vae, X_train, epochs, batch_size, validation_split=0.1):
    """
    Treina o modelo VAE com os dados de treinamento.
    Args:
        vae (keras.Model): O modelo VAE construído.
        X_train (np.array): Dados de treinamento.
        epochs (int): Número de épocas de treinamento.
        batch_size (int): Tamanho do batch para o treinamento.
        validation_split (float): Fração dos dados de treinamento a ser usada como conjunto de validação.
    Returns:
        keras.callbacks.History: Objeto contendo o histórico do treinamento.
    """
    history = vae.fit(X_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

# ------------------------------------------------------------------------------
# 8. Detecção de Anomalias com o VAE
# ------------------------------------------------------------------------------

def detectar_anomalias(vae, X_test, threshold_percentile=95):
    """
    Detecta anomalias nos dados de teste usando o erro de reconstrução do VAE.
    Args:
        vae (keras.Model): O modelo VAE treinado.
        X_test (np.array): Dados de teste.
        threshold_percentile (float): Percentil do erro de reconstrução para definir o limiar de anomalia.
    Returns:
        tuple: Uma tupla contendo os erros de reconstrução e um array booleano indicando as anomalias.
    """
    reconstructions = vae.predict(X_test)
    mse = np.mean(np.square(X_test - reconstructions), axis=(1, 2, 3))
    threshold = np.percentile(mse, threshold_percentile)
    anomalies = mse > threshold
    return mse, anomalies

# ------------------------------------------------------------------------------
# 9. Visualização das Anomalias (Opcional)
# ------------------------------------------------------------------------------

def visualizar_anomalias(recurrence_plots_original, indices_anomalos, num_visualizar=5):
    """
    Visualiza os recurrence plots considerados anômalos.
    Args:
        recurrence_plots_original (list): Lista dos RPs originais.
        indices_anomalos (np.array): Índices dos RPs considerados anômalos.
        num_visualizar (int): Número de anomalias para visualizar.
    """
    import matplotlib.pyplot as plt

    def visualizar_recurrence_plot(rp, title="Recurrence Plot"):
        plt.imshow(rp, cmap='binary', origin='lower')
        plt.title(title)
        plt.xlabel('Tempo')
        plt.ylabel('Tempo')
        plt.colorbar(label='Recorrência')
        plt.show()

    print("\nExemplos de Recurrence Plots Anômalos:")
    anomalies_to_visualize = np.array(recurrence_plots_original)[indices_anomalos]
    for i in range(min(num_visualizar, len(anomalies_to_visualize))):
        visualizar_recurrence_plot(anomalies_to_visualize[i], title=f"Anomalia {indices_anomalos[i]}")


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
        arquivo_orderbook = "D:\Temp\Lobster data\AMZN_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\AAPL_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\INTC_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\GOOG_2012-06-21_34200000_57600000_orderbook_1.csv"
        #arquivo_orderbook = "D:\Temp\Lobster data\MSFT_2012-06-21_34200000_57600000_orderbook_1.csv"
        tamanho_janela = 1000
        passo = 500
        time_delay = 1
        dimension = 1
        threshold = 'distance'
        percentage = 10
        num_graficos_para_visualizar = 5
        latent_dim = 32
        epochs = 50
        batch_size = 32
        threshold_percentile = 95

        # Abordagem com livro de ofertas simplificado
        print("Carregando o livro de ofertas...")
        df_livro_ordens = carregar_livro_ordens(arquivo_orderbook)
        print("Livro de ofertas carregado com sucesso.")
        
        print(df_livro_ordens.head())

        print("\nCriando janelas deslizantes...")
        colunas_para_janelas = ['best_ask', 'best_bid']
        #colunas_para_janelas = ['best_ask']
        dados_para_janelas = df_livro_ordens[colunas_para_janelas]
        janelas_deslizantes = criar_janelas_deslizantes(dados_para_janelas, tamanho_janela, passo)
        print(f"{len(janelas_deslizantes)} janelas deslizantes criadas.")

        print("\nGerando recurrence plots...")
        recurrence_plots = gerar_recurrence_plots(janelas_deslizantes, time_delay, dimension, threshold, percentage)
        print(f"{len(recurrence_plots)} recurrence plots gerados.")

        if recurrence_plots:
            #print("\nVisualizando os recurrence plots...")
            #visualizar_recurrence_plots(recurrence_plots, num_graficos_para_visualizar)

            print("\nPreparando dados para o VAE...")
            X_train, X_test, input_shape = preparar_dados_vae(recurrence_plots)

            print("\nConstruindo o VAE...")
            encoder, decoder, vae = construir_vae(input_shape, latent_dim)
            vae.summary()

            print("\nTreinando o VAE...")
            history = treinar_vae(vae, X_train, epochs, batch_size)

            print("\nDetectando anomalias com o VAE...")
            mse, anomalies = detectar_anomalias(vae, X_test, threshold_percentile)
            indices_anomalos = np.where(anomalies)[0]
            print(f"Número de anomalias detectadas: {len(indices_anomalos)}")

            print("\nVisualizando os recurrence plots...")
            visualizar_recurrence_plots(recurrence_plots, num_graficos_para_visualizar)

            print("\nVisualizando as anomalias detectadas (amostra)...")
            visualizar_anomalias(recurrence_plots, indices_anomalos)
        else:
            print("\nNenhum recurrence plot foi gerado.")

    except FileNotFoundError as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()