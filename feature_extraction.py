from pathlib import Path
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


def processar_dataset(base_dir, nome_dataset, cabecalho):
    pasta_dataset = base_dir / nome_dataset
    pastas_classes = {
        'nominal': 0,
        'falha': 1
    }

    lista_dfs = []
    rotulos = {}
    global_id = 0

    print(f"\nProcessando arquivos CSV de '{nome_dataset}'...")
    for nome_pasta, classe in pastas_classes.items():
        pasta = pasta_dataset / nome_pasta
        if not pasta.exists():
            raise FileNotFoundError(f"Pasta de dados não encontrada: '{pasta}'.")

        arquivos = sorted(list(pasta.glob('*.csv')))
        if not arquivos:
            raise ValueError(f"Nenhum arquivo .csv encontrado em '{pasta}'.")
        print(f"Carregando {len(arquivos)} arquivo(s) de '{pasta}' (Classe {classe})...")

        for arquivo in arquivos:
            df_sim = pd.read_csv(arquivo, header=None, names=cabecalho)
            df_sim.rename(columns={'tempo': 'time'}, inplace=True)

            df_sim['id'] = global_id

            lista_dfs.append(df_sim)
            rotulos[global_id] = classe

            global_id += 1

    if not lista_dfs:
        raise ValueError(f"Nenhum arquivo .csv foi carregado de '{nome_dataset}'.")

    df_tsfresh = pd.concat(lista_dfs, ignore_index=True)
    y = pd.Series(rotulos, name='classe')

    print(f"Leitura concluída. Total de simulações processadas: {global_id}")

    print(f"Extraindo features de '{nome_dataset}' com TSFresh...")
    X_features = extract_features(
        df_tsfresh,
        column_id='id',
        column_sort='time'
    )

    X_features = impute(X_features)

    arquivo_parquet = pasta_dataset / 'X_features_tsfresh.parquet'
    arquivo_csv = pasta_dataset / 'y_labels.csv'

    X_features.to_parquet(arquivo_parquet)
    y.to_csv(arquivo_csv, index_label='id')

    print(f"- Matriz de features salva em: '{arquivo_parquet}'")
    print(f"- Rótulos salvos em:          '{arquivo_csv}'")


def main():
    base_dir = Path(__file__).resolve().parent
    cabecalho = [
        'tempo', 'C_im', 'T_im', 'T_cim', 'Ci', 'Ti',
        'C', 'T', 'Qc', 'Tci', 'Tc'
    ]
    for nome_dataset in ('training_dataset', 'test_dataset'):
        processar_dataset(base_dir, nome_dataset, cabecalho)

    print("\n" + "="*50)
    print("PROCESSAMENTO DE TREINO E TESTE CONCLUÍDO COM SUCESSO!")
    print("="*50)


if __name__ == '__main__':
    main()