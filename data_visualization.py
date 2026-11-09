from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CABECALHO = [
    'tempo', 'C_im', 'T_im', 'T_cim', 'Ci', 'Ti',
    'C', 'T', 'Qc', 'Tci', 'Tc'
]
VARIAVEIS = CABECALHO[1:]
DURACAO_RECORTE_MINUTOS = 50


def salvar_graficos(arquivo, diretorio_saida, identificacao):
    dados = pd.read_csv(arquivo, header=None, names=CABECALHO)
    tempo = dados['tempo']
    intervalos = {
        'tempo_completo': dados,
        'primeiros_50_minutos': dados[
            tempo <= tempo.iloc[0] + DURACAO_RECORTE_MINUTOS
        ],
    }

    for nome_intervalo, dados_intervalo in intervalos.items():
        pasta_intervalo = diretorio_saida / nome_intervalo
        pasta_intervalo.mkdir(parents=True, exist_ok=True)

        for variavel in VARIAVEIS:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(dados_intervalo['tempo'], dados_intervalo[variavel], linewidth=1.2)
            ax.set_title(f'{identificacao} - {variavel} ({nome_intervalo.replace("_", " ")})')
            ax.set_xlabel('Tempo (min)')
            ax.set_ylabel(variavel)
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()
            fig.savefig(pasta_intervalo / f'{variavel}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    imagens_dir = base_dir / 'images' / 'visualizacao_base'

    arquivos_nominais = sorted((base_dir / 'training_dataset' / 'nominal').glob('*.csv'))
    if not arquivos_nominais:
        raise FileNotFoundError('Nenhum CSV encontrado em training_dataset/nominal.')

    arquivos_falha = sorted((base_dir / 'training_dataset' / 'falha').glob('*.csv'))
    if len(arquivos_falha) != 600:
        raise ValueError(
            'Esperados 600 CSVs em training_dataset/falha, organizados em três grupos de 200.'
        )

    arquivo_nominal = arquivos_nominais[0]
    salvar_graficos(
        arquivo_nominal,
        imagens_dir / 'nominal' / arquivo_nominal.stem,
        f'Nominal - {arquivo_nominal.stem}',
    )

    grupos_falha = {
        'sutil': arquivos_falha[0],
        'media': arquivos_falha[200],
        'severa': arquivos_falha[400],
    }
    for intensidade, arquivo in grupos_falha.items():
        salvar_graficos(
            arquivo,
            imagens_dir / 'falha' / intensidade / arquivo.stem,
            f'Falha {intensidade} - {arquivo.stem}',
        )

    print(f'Gráficos salvos em: {imagens_dir}')


if __name__ == '__main__':
    main()
