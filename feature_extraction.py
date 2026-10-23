from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# Define o diretório de trabalho
os.chdir(r"C:\Users\felip\Documents\TCC_2nd_batch")

def main():
    # ---------------------------------------------------------
    # 1. Configurações Globais e Mapeamento das Classes
    # ---------------------------------------------------------
    colunas_vars = [
        'Ci', 'Ti', 'Tci', 'Ci_ruido', 'Ti_ruido', 
        'C', 'T', 'Qc', 'Tci_ruido', 'Tc'
    ]

    # Mapeamento atualizado com os nomes corretos das pastas
    pastas_classes = {
        r'test_dataset\nominal': 0,
        r'test_dataset\falha': 1
    }

    lista_dfs = []
    rotulos = {}
    info_simulacoes = []
    global_id = 0

    # ---------------------------------------------------------
    # 2. Leitura dos Arquivos CSV e Montagem do DataFrame
    # ---------------------------------------------------------
    print("Iniciando leitura dos arquivos .csv...")

    for pasta_caminho, classe in pastas_classes.items():
        pasta = Path(pasta_caminho)
        if not pasta.exists():
            print(f"Aviso: Pasta '{pasta_caminho}' não encontrada. Verifique o caminho.")
            continue

        arquivos = sorted(list(pasta.glob('*.csv')))
        print(f"Carregando {len(arquivos)} arquivo(s) de '{pasta_caminho}' (Classe {classe})...")

        for arquivo in arquivos:
            try:
                df_raw = pd.read_csv(arquivo, header=None)
                
                # Se tiver 11 colunas (1 tempo + 10 variáveis)
                if df_raw.shape[1] == len(colunas_vars) + 1:
                    df_sim = pd.DataFrame()
                    df_sim['time'] = df_raw.iloc[:, 0]
                    for idx, col_nome in enumerate(colunas_vars):
                        df_sim[col_nome] = df_raw.iloc[:, idx + 1]
                
                # Se tiver 10 colunas (apenas variáveis)
                elif df_raw.shape[1] == len(colunas_vars):
                    df_sim = pd.DataFrame(df_raw.values, columns=colunas_vars)
                    df_sim['time'] = np.arange(len(df_sim))
                
                # Se já tiver cabeçalho
                else:
                    df_sim = pd.read_csv(arquivo)
                    if 'time' not in df_sim.columns and 'tempo' in df_sim.columns:
                        df_sim.rename(columns={'tempo': 'time'}, inplace=True)

                df_sim['id'] = global_id
                
                lista_dfs.append(df_sim)
                rotulos[global_id] = classe
                
                info_simulacoes.append({
                    'id': global_id,
                    'arquivo': arquivo.name,
                    'classe': classe
                })
                
                global_id += 1

            except Exception as e:
                print(f"Erro ao processar arquivo '{arquivo.name}': {e}")

    if not lista_dfs:
        raise ValueError("Nenhum arquivo .csv foi carregado. Verifique os caminhos no dicionário 'pastas_classes'.")

    df_tsfresh = pd.concat(lista_dfs, ignore_index=True)
    y = pd.Series(rotulos, name='classe')

    print(f"Leitura concluída. Total de simulações processadas: {global_id}")

    # ---------------------------------------------------------
    # 3. Gráficos de Verificação (Visualização e Salvamento)
    # ---------------------------------------------------------
    print("Gerando gráfico de verificação da leitura...")
    df_info = pd.DataFrame(info_simulacoes)
    ids_para_plot = []

    for c in set(rotulos.values()):
        ids_para_plot.extend(df_info[df_info['classe'] == c]['id'].head(2).tolist())

    vars_para_plot = ['C', 'T', 'Qc', 'Ci_ruido']
    fig, axes = plt.subplots(len(vars_para_plot), 1, figsize=(12, 10), sharex=True)

    if len(vars_para_plot) == 1:
        axes = [axes]

    for ax, var in zip(axes, vars_para_plot):
        for sim_id in ids_para_plot:
            dados_sim = df_tsfresh[df_tsfresh['id'] == sim_id]
            classe_sim = rotulos[sim_id]
            ax.plot(
                dados_sim['time'], 
                dados_sim[var], 
                label=f'ID {sim_id} (Classe {classe_sim})',
                linewidth=1.2
            )
        ax.set_ylabel(var)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Tempo (min)')
    fig.suptitle('Verificação de Leitura dos Dados (CSV) - Amostras Selecionadas', fontsize=14)
    plt.tight_layout()

    nome_imagem = 'images/leitura_test.png'
    plt.savefig(nome_imagem, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo com sucesso como '{nome_imagem}'.")

    # ---------------------------------------------------------
    # 4. Extração de Features com TSFresh e Persistência
    # ---------------------------------------------------------
    print("Extraindo features com TSFresh...")
    X_features = extract_features(
        df_tsfresh, 
        column_id='id', 
        column_sort='time'
    )

    X_features = impute(X_features)

    arquivo_parquet = 'test_dataset/X_features_tsfresh.parquet'
    arquivo_csv = 'test_dataset/y_labels.csv'

    X_features.to_parquet(arquivo_parquet)
    y.to_csv(arquivo_csv, index_label='id')

    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"- Matriz de features salva em: '{arquivo_parquet}'")
    print(f"- Rótulos salvos em:          '{arquivo_csv}'")
    print(f"- Imagem de verificação:      '{nome_imagem}'")
    print("="*50)


if __name__ == '__main__':
    main()