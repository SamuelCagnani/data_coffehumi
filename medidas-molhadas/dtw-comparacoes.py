#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dtw-comparacoes.py
==================

Objetivo
--------
Comparar as series temporais de corrente (A) usando DTW
(Dynamic Time Warping, ou Distancia de Deformacao Temporal).

O que e DTW?
------------
Duas series temporais podem representar o mesmo fenomeno fisico, mas com
pequenos atrasos ou duracoes diferentes. A distancia euclidiana ponto a ponto
compara sempre o instante t da serie A com o instante t da serie B. Se um
evento acontece 0,2 s antes em um dos ensaios, a euclidiana "ve" uma grande
diferenca, mesmo que as curvas tenham o mesmo formato.

O DTW resolve isso procurando o MELHOR ALINHAMENTO entre as duas series:
ele pode "esticar" ou "encolher" o eixo do tempo de uma delas. A distancia
final e a raiz da soma das diferencas ao quadrado ao longo do caminho otimo:

    DTW(A, B) = sqrt( soma de (A[i] - B[j])^2 para cada par (i, j) do caminho )

Quanto menor o valor, mais parecidas sao as formas das curvas. O DTW NAO
exige que as duas series tenham o mesmo numero de amostras.

O que este script calcula?
--------------------------
1) REPETIBILIDADE
   Compara, com DTW, todos os pares de ensaios do MESMO peso. Se os valores
   forem baixos, significa que repetir o ensaio naquele peso produz curvas
   parecidas (boa repetibilidade).

2) COMPARACAO ENTRE PESOS
   Primeiro calcula a serie media de cada peso (media ponto a ponto dos
   ensaios daquele peso). Depois compara as medias dos pesos entre si com
   DTW e com a distancia euclidiana. A razao DTW/euclidiana mostra quanto o
   alinhamento no tempo "economizou" de distancia.

3) CAMINHO DE ALINHAMENTO
   Para o par de pesos mais diferente, desenha a matriz de custo e o caminho
   escolhido pelo DTW. Isso ajuda a visualizar quais amostras foram ligadas.

Como executar
-------------
    python dtw-comparacoes.py

Dependencias: numpy, pandas, matplotlib e tslearn.

Arquivos gerados (na mesma pasta deste script)
----------------------------------------------
    dtw-repetibilidade-<seca|molhada>.csv
    dtw-repetibilidade-<seca|molhada>.png
    dtw-entre-pesos-<seca|molhada>.csv
    dtw-matriz-<seca|molhada>.png
    dtw-alinhamento-<seca|molhada>.png
"""

from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# "Agg" apenas desenha e salva a imagem, sem abrir janela. Isso permite rodar
# o script em qualquer maquina, inclusive em servidores sem tela.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# O tslearn concentra as funcoes de DTW. Se ele nao estiver instalado,
# mostramos uma mensagem amigavel em vez de um erro enorme de importacao.
try:
    from tslearn.metrics import dtw, dtw_path
except ModuleNotFoundError as erro:
    raise SystemExit(
        "A biblioteca 'tslearn' nao esta instalada.\n"
        "Instale com:\n"
        "    uv pip install --python .venv/bin/python tslearn==0.10.0"
    ) from erro

# Garante que acentos apareçam corretamente no terminal do Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Configuracoes (os "numeros magicos" ficam todos aqui)
# ---------------------------------------------------------------------------
PASTA = Path(__file__).resolve().parent   # pasta onde este script esta salvo
PADRAO_CSV = "*.csv"                      # quais arquivos serao lidos
SEPARADOR = ","                           # separador das colunas do CSV
COLUNA_TEMPO = "Tempo"                    # nome da coluna de tempo
COLUNA_CORRENTE = "Corrente"              # nome da coluna de corrente
FORMATO_TEMPO = "%H:%M:%S.%f"             # hora:minuto:segundo.milissegundo
PASSO_GRADE_S = 0.05                      # passo da grade comum de tempo (s)
DPI = 150                                 # resolucao das imagens salvas


# ---------------------------------------------------------------------------
# Funcoes auxiliares de leitura e nomes
# ---------------------------------------------------------------------------
def sufixo_figura(pasta: Path) -> str:
    """Sufixo usado nos nomes dos arquivos: 'seca' ou 'molhada'."""
    nome = pasta.name.lower()
    if "molhada" in nome:
        return "molhada"
    if "seca" in nome:
        return "seca"
    return re.sub(r"\W+", "-", pasta.name)


def nome_amigavel(pasta: Path) -> str:
    """Titulo legivel para os graficos."""
    if "molhada" in pasta.name.lower():
        return "Pedra Molhada"
    if "seca" in pasta.name.lower():
        return "Pedra Seca"
    return pasta.name


def rotulo_peso(peso: int) -> str:
    """Transforma 0 em 'vazio' para os rotulos ficarem mais claros."""
    return "vazio" if peso == 0 else f"{peso} kg"


def extrair_peso(nome_arquivo: str) -> int | None:
    """Extrai o peso (kg) do nome do arquivo.

    Exemplos:
        '220kgPedraMolhada.csv'    -> 220
        '220kgPedraMolhada(2).csv' -> 220
        '0kgPedraSeca(5).csv'      -> 0

    Retorna None quando o nome nao segue o padrao '<numero>kg'.
    """
    encontrado = re.search(r"(\d+)\s*kg", nome_arquivo, flags=re.IGNORECASE)
    return int(encontrado.group(1)) if encontrado else None


def carregar_ensaio(caminho: Path) -> pd.DataFrame:
    """Le um CSV e devolve 'tempo_s' (segundos decorridos) e 'Corrente'.

    A coluna original 'Tempo' vem no formato HH:MM:SS.mmm. Subtraindo o
    primeiro instante de todos os outros, cada ensaio comeca em t = 0.
    Isso e importante para comparar ensaios que comecaram em horas diferentes.
    """
    df = pd.read_csv(caminho, sep=SEPARADOR)

    faltando = [c for c in (COLUNA_TEMPO, COLUNA_CORRENTE) if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas {faltando} nao encontradas em '{caminho.name}'.")

    instantes = pd.to_datetime(df[COLUNA_TEMPO], format=FORMATO_TEMPO)
    df["tempo_s"] = (instantes - instantes.iloc[0]).dt.total_seconds()

    return df[["tempo_s", COLUNA_CORRENTE]].copy()


def carregar_grupos(arquivos: list[Path]) -> dict[int, list[tuple[str, pd.DataFrame]]]:
    """Le todos os CSV e agrupa os ensaios por peso.

    Estrutura devolvida:
        {
            0:   [("0kgPedraSeca.csv", DataFrame), ("0kgPedraSeca(2).csv", DataFrame), ...],
            100: [("100kgPedraSeca.csv", DataFrame), ...],
            ...
        }
    """
    grupos: dict[int, list[tuple[str, pd.DataFrame]]] = {}

    for caminho in arquivos:
        peso = extrair_peso(caminho.name)
        if peso is None:
            print(f"[aviso] ignorando arquivo sem peso no nome: {caminho.name}")
            continue
        grupos.setdefault(peso, []).append((caminho.name, carregar_ensaio(caminho)))

    return grupos


def interpolar_na_grade(ensaio: pd.DataFrame, grade: np.ndarray) -> np.ndarray:
    """Reprojeta um ensaio na grade de tempo comum por interpolacao linear.

    Fora do intervalo coberto pelo ensaio o valor e NaN (vazio). Assim, ao
    empilhar varios ensaios, regioes sem dado nao contaminam a media.
    """
    return np.interp(
        grade,
        ensaio["tempo_s"].to_numpy(),
        ensaio[COLUNA_CORRENTE].to_numpy(),
        left=np.nan,
        right=np.nan,
    )


# ---------------------------------------------------------------------------
# 1) Repetibilidade: DTW entre ensaios do mesmo peso
# ---------------------------------------------------------------------------
def calcular_repetibilidade(
    grupos: dict[int, list[tuple[str, pd.DataFrame]]]
) -> list[dict]:
    """Calcula DTW para todos os pares de ensaios de um mesmo peso.

    Usamos as series BRUTAS (sem interpolar), porque o DTW aceita series de
    comprimentos diferentes. Cada linha do resultado representa um par:

        peso, ensaio_a, ensaio_b, distancia_dtw
    """
    linhas: list[dict] = []

    for peso in sorted(grupos):
        ensaios = grupos[peso]

        # Com apenas um ensaio nao existe par para comparar.
        if len(ensaios) < 2:
            print(
                f"[aviso] {rotulo_peso(peso)}: apenas {len(ensaios)} ensaio(s); "
                "nao ha pares para repetibilidade."
            )
            continue

        # combinations(ensaios, 2) gera todos os pares (A, B) sem repetir
        # a mesma comparacao e sem comparar um ensaio com ele mesmo.
        for (nome_a, df_a), (nome_b, df_b) in combinations(ensaios, 2):
            serie_a = df_a[COLUNA_CORRENTE].to_numpy(dtype=float)
            serie_b = df_b[COLUNA_CORRENTE].to_numpy(dtype=float)

            # dtw(...) devolve a distancia DTW (quanto menor, mais parecido).
            distancia = float(dtw(serie_a, serie_b))

            linhas.append(
                {
                    "peso": peso,
                    "ensaio_a": nome_a,
                    "ensaio_b": nome_b,
                    "distancia_dtw": distancia,
                }
            )

    return linhas


def resumir_repetibilidade(linhas: list[dict]) -> pd.DataFrame:
    """Agrupa as distancias de repetibilidade por peso.

    Para cada peso calculamos: numero de pares, media, desvio padrao,
    menor e maior DTW. O desvio padrao mostra se a repetibilidade e
    uniforme entre os pares daquele peso.
    """
    if not linhas:
        return pd.DataFrame()

    tabela = pd.DataFrame(linhas)

    resumo = (
        tabela.groupby("peso")["distancia_dtw"]
        .agg(
            n_pares="count",
            dtw_media="mean",
            dtw_desvio="std",
            dtw_min="min",
            dtw_max="max",
        )
        .reset_index()
    )

    return resumo


# ---------------------------------------------------------------------------
# 2) Serie media de cada peso em uma grade de tempo comum
# ---------------------------------------------------------------------------
def calcular_series_medias(
    grupos: dict[int, list[tuple[str, pd.DataFrame]]]
) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    """Calcula a serie media de cada peso.

    Como os ensaios tem duracoes diferentes, escolhemos uma JANELA COMUM:
    de 0 ate a MENOR duracao entre todos os ensaios da pasta. Assim todos
    os ensaios contribuem em todos os instantes da grade, e a media e valida
    do inicio ao fim.

    Retorna:
        grade  -> vetor de tempos (s)
        medias -> dicionario {peso: vetor de correntes medias}
        janela -> duracao da janela comum (s)
    """
    duracoes = [
        df["tempo_s"].iloc[-1]
        for ensaios in grupos.values()
        for _, df in ensaios
    ]
    janela = float(min(duracoes))

    # Grade regular: 0, 0.05, 0.10, ... ate pouco antes de 'janela'.
    grade = np.arange(0.0, janela, PASSO_GRADE_S)

    medias: dict[int, np.ndarray] = {}

    for peso in sorted(grupos):
        # Empilha os ensaios do peso: uma linha por ensaio, uma coluna por
        # instante da grade. O formato final e [n_ensaios x n_instantes].
        matriz = np.vstack(
            [interpolar_na_grade(df, grade) for _, df in grupos[peso]]
        )

        # Media ponto a ponto (ignorando NaN, caso exista algum buraco).
        medias[peso] = np.nanmean(matriz, axis=0)

    return grade, medias, janela


# ---------------------------------------------------------------------------
# 3) DTW entre as series medias dos diferentes pesos
# ---------------------------------------------------------------------------
def calcular_entre_pesos(
    medias: dict[int, np.ndarray]
) -> tuple[list[dict], np.ndarray, list[int]]:
    """Compara as series medias dos pesos dois a dois.

    Para cada par (peso_a, peso_b) calculamos:
        - distancia_dtw: DTW entre as series medias;
        - distancia_euclidiana: norma L2 ponto a ponto;
        - razao_dtw_euclidiana: DTW / euclidiana.

    Como as duas series medias estao na MESMA grade, a euclidiana e uma
    comparacao justa. Uma razao menor que 1 indica que o DTW encontrou um
    alinhamento melhor do que a comparacao ponto a ponto.

    Retorna tambem a matriz DTW completa (simetrica, diagonal zero) e a
    lista de pesos na ordem usada nas linhas/colunas da matriz.
    """
    pesos = sorted(medias)
    linhas: list[dict] = []
    matriz = np.zeros((len(pesos), len(pesos)), dtype=float)

    # Percorremos apenas j > i para calcular cada par uma unica vez.
    for i, peso_a in enumerate(pesos):
        for j in range(i + 1, len(pesos)):
            peso_b = pesos[j]

            serie_a = medias[peso_a]
            serie_b = medias[peso_b]

            distancia_dtw = float(dtw(serie_a, serie_b))

            # A matriz e simetrica: DTW(A, B) = DTW(B, A).
            matriz[i, j] = distancia_dtw
            matriz[j, i] = distancia_dtw

            # Distancia euclidiana entre as duas series medias.
            distancia_euclidiana = float(np.linalg.norm(serie_a - serie_b))

            if distancia_euclidiana > 0:
                razao = distancia_dtw / distancia_euclidiana
            else:
                razao = float("nan")

            linhas.append(
                {
                    "peso_a": peso_a,
                    "peso_b": peso_b,
                    "distancia_dtw": distancia_dtw,
                    "distancia_euclidiana": distancia_euclidiana,
                    "razao_dtw_euclidiana": razao,
                }
            )

    return linhas, matriz, pesos


# ---------------------------------------------------------------------------
# 4) Graficos
# ---------------------------------------------------------------------------
def salvar_csv(linhas: list[dict], caminho: Path) -> None:
    """Salva uma lista de dicionarios como CSV."""
    pd.DataFrame(linhas).to_csv(caminho, index=False, float_format="%.6f")


def plotar_repetibilidade(
    linhas: list[dict], resumo: pd.DataFrame, caminho: Path, titulo: str
) -> None:
    """Grafico de barras da DTW media por peso, com os pares individuais.

    A barra mostra a media; a barra de erro mostra o desvio padrao; os
    pontos pretos mostram cada par de ensaios. Assim da para ver tanto o
    comportamento medio quanto a dispersao.
    """
    if resumo.empty:
        print("[aviso] sem pares de repetibilidade; grafico nao gerado.")
        return

    tabela = pd.DataFrame(linhas)
    posicoes = np.arange(len(resumo))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        posicoes,
        resumo["dtw_media"],
        yerr=resumo["dtw_desvio"].fillna(0.0),
        capsize=5,
        color="#4C72B0",
        edgecolor="black",
        alpha=0.85,
        label="DTW media por peso",
    )

    # Desenha cada par individualmente. O pequeno deslocamento horizontal
    # (jitter) evita que pontos com o mesmo valor fiquem exatamente sobrepostos.
    for posicao, peso in zip(posicoes, resumo["peso"]):
        valores = tabela.loc[tabela["peso"] == peso, "distancia_dtw"].to_numpy()
        if len(valores) > 1:
            jitter = np.linspace(-0.08, 0.08, len(valores))
        else:
            jitter = np.zeros(1)
        ax.scatter(
            np.full(len(valores), posicao) + jitter,
            valores,
            color="black",
            s=22,
            zorder=3,
            label="Pares individuais" if posicao == posicoes[0] else None,
        )

    ax.set_xticks(posicoes)
    ax.set_xticklabels([rotulo_peso(p) for p in resumo["peso"]])
    ax.set_title(titulo)
    ax.set_xlabel("Peso")
    ax.set_ylabel("Distancia DTW (A)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(caminho, dpi=DPI)
    plt.close(fig)


def plotar_matriz_dtw(
    pesos: list[int],
    matriz: np.ndarray,
    caminho: Path,
    titulo: str,
) -> None:
    """Heatmap da matriz DTW entre as series medias dos pesos.

    Cada celula (linha = peso A, coluna = peso B) mostra a distancia DTW.
    A diagonal principal e zero, porque a distancia de uma serie com ela
    mesma e zero.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # vmin=0 fixa a escala de cores comecando em zero, facilitando comparar
    # esta matriz com outras geradas pelo script.
    imagem = ax.imshow(matriz, cmap="viridis", vmin=0.0)
    fig.colorbar(imagem, ax=ax, label="Distancia DTW (A)")

    rotulos = [rotulo_peso(p) for p in pesos]
    ax.set_xticks(range(len(pesos)))
    ax.set_xticklabels(rotulos, rotation=45, ha="right")
    ax.set_yticks(range(len(pesos)))
    ax.set_yticklabels(rotulos)
    ax.set_xlabel("Peso")
    ax.set_ylabel("Peso")
    ax.set_title(titulo)

    # Escreve o valor numerico em cada celula. A cor do texto muda para
    # continuar legivel tanto nas celulas escuras quanto nas claras.
    maior_valor = float(np.nanmax(matriz)) if matriz.size else 0.0
    for i in range(len(pesos)):
        for j in range(len(pesos)):
            cor_texto = "white" if matriz[i, j] > 0.6 * maior_valor else "black"
            ax.text(
                j,
                i,
                f"{matriz[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=cor_texto,
            )

    fig.tight_layout()
    fig.savefig(caminho, dpi=DPI)
    plt.close(fig)


def plotar_alinhamento(
    grade: np.ndarray,
    medias: dict[int, np.ndarray],
    linhas_entre_pesos: list[dict],
    caminho: Path,
    titulo: str,
) -> None:
    """Mostra o caminho de alinhamento do par de pesos mais diferente.

    A figura tem duas partes:
      - em cima: as duas series medias, para lembrar o formato das curvas;
      - embaixo: a matriz de custo local e o caminho escolhido pelo DTW.
        Cada ponto (i, j) do caminho significa que a amostra i da serie A
        foi alinhada com a amostra j da serie B.
    """
    if not linhas_entre_pesos:
        print("[aviso] sem pares entre pesos; grafico de alinhamento nao gerado.")
        return

    # Escolhe o par com maior DTW: normalmente e o caso em que o alinhamento
    # deforma mais o tempo e, portanto, o mais interessante de visualizar.
    par = max(linhas_entre_pesos, key=lambda linha: linha["distancia_dtw"])
    peso_a = par["peso_a"]
    peso_b = par["peso_b"]

    serie_a = medias[peso_a]
    serie_b = medias[peso_b]

    # dtw_path devolve:
    #   caminho -> lista de pares (i, j) do alinhamento otimo
    #   distancia -> mesma distancia DTW calculada por dtw(...)
    caminho_dtw, distancia = dtw_path(serie_a, serie_b)

    fig, (ax_topo, ax_baixo) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1.4]}
    )

    # --- Parte de cima: as duas series medias -----------------------------
    ax_topo.plot(
        grade,
        serie_a,
        color="#4C72B0",
        linewidth=1.8,
        label=rotulo_peso(peso_a),
    )
    ax_topo.plot(
        grade,
        serie_b,
        color="#DD8452",
        linewidth=1.8,
        linestyle="--",
        label=rotulo_peso(peso_b),
    )
    ax_topo.set_title(
        f"Alinhamento DTW: {rotulo_peso(peso_a)} x {rotulo_peso(peso_b)} "
        f"(DTW = {float(distancia):.2f} A)"
    )
    ax_topo.set_xlabel("Tempo decorrido (s)")
    ax_topo.set_ylabel("Corrente media (A)")
    ax_topo.grid(True, linestyle="--", alpha=0.4)
    ax_topo.legend(fontsize=9)

    # --- Parte de baixo: matriz de custo + caminho ------------------------
    # Custo local: diferenca ao quadrado entre cada amostra de A e de B.
    custo = (serie_a[:, None] - serie_b[None, :]) ** 2

    # Transpoe para o eixo x representar as amostras de A e o eixo y as de B.
    ax_baixo.imshow(custo.T, origin="lower", aspect="auto", cmap="magma")
    ax_baixo.plot(
        [i for i, _ in caminho_dtw],
        [j for _, j in caminho_dtw],
        color="cyan",
        linewidth=1.5,
        label="Caminho otimo",
    )
    ax_baixo.set_title("Matriz de custo local e caminho escolhido pelo DTW")
    ax_baixo.set_xlabel(f"Amostras de {rotulo_peso(peso_a)}")
    ax_baixo.set_ylabel(f"Amostras de {rotulo_peso(peso_b)}")
    ax_baixo.legend(fontsize=9)

    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(caminho, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    arquivos = sorted(PASTA.glob(PADRAO_CSV))
    if not arquivos:
        raise SystemExit(f"Nenhum arquivo CSV encontrado em: {PASTA}")

    grupos = carregar_grupos(arquivos)
    if not grupos:
        raise SystemExit("Nenhum arquivo com peso (padrao '<numero>kg') foi encontrado.")

    sufixo = sufixo_figura(PASTA)
    regiao = nome_amigavel(PASTA)

    print("=" * 70)
    print(f"Pasta analisada : {PASTA}")
    print(f"Arquivos lidos  : {len(arquivos)}")
    print(f"Pesos encontrados: {[rotulo_peso(p) for p in sorted(grupos)]}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Repetibilidade dentro de cada peso
    # ------------------------------------------------------------------
    print("\n[1] Repetibilidade: DTW entre ensaios do mesmo peso")
    print("-" * 70)

    linhas_repetibilidade = calcular_repetibilidade(grupos)
    resumo_repetibilidade = resumir_repetibilidade(linhas_repetibilidade)

    if linhas_repetibilidade:
        arq_csv = PASTA / f"dtw-repetibilidade-{sufixo}.csv"
        arq_png = PASTA / f"dtw-repetibilidade-{sufixo}.png"

        salvar_csv(linhas_repetibilidade, arq_csv)
        plotar_repetibilidade(
            linhas_repetibilidade,
            resumo_repetibilidade,
            arq_png,
            f"Repetibilidade dos ensaios - {regiao}",
        )

        for _, linha in resumo_repetibilidade.iterrows():
            desvio = linha["dtw_desvio"]
            desvio_texto = "0.00" if pd.isna(desvio) else f"{desvio:.2f}"
            print(
                f"  {rotulo_peso(int(linha['peso'])):>8} | "
                f"{int(linha['n_pares']):2d} par(es) | "
                f"DTW media {linha['dtw_media']:6.2f} | "
                f"desvio {desvio_texto} | "
                f"faixa {linha['dtw_min']:.2f} a {linha['dtw_max']:.2f}"
            )

        print(f"  CSV salvo : {arq_csv}")
        print(f"  PNG salvo : {arq_png}")
    else:
        print("  Nenhum par de repetibilidade foi calculado.")

    # ------------------------------------------------------------------
    # 2) Series medias de cada peso
    # ------------------------------------------------------------------
    print("\n[2] Serie media de cada peso")
    print("-" * 70)

    grade, medias, janela = calcular_series_medias(grupos)
    print(f"  Janela comum: 0 a {janela:.1f} s ({grade.size} pontos)")
    for peso in sorted(medias):
        print(
            f"  {rotulo_peso(peso):>8} | "
            f"corrente media {np.nanmean(medias[peso]):5.2f} A"
        )

    # ------------------------------------------------------------------
    # 3) DTW entre as series medias dos diferentes pesos
    # ------------------------------------------------------------------
    print("\n[3] DTW entre as series medias dos pesos")
    print("-" * 70)

    linhas_entre_pesos, matriz, pesos = calcular_entre_pesos(medias)

    if linhas_entre_pesos:
        arq_csv = PASTA / f"dtw-entre-pesos-{sufixo}.csv"
        arq_png = PASTA / f"dtw-matriz-{sufixo}.png"

        salvar_csv(linhas_entre_pesos, arq_csv)
        plotar_matriz_dtw(
            pesos,
            matriz,
            arq_png,
            f"Matriz DTW entre as series medias - {regiao}",
        )

        for linha in linhas_entre_pesos:
            print(
                f"  {rotulo_peso(linha['peso_a']):>8} x "
                f"{rotulo_peso(linha['peso_b']):<8} | "
                f"DTW {linha['distancia_dtw']:7.2f} | "
                f"euclidiana {linha['distancia_euclidiana']:7.2f} | "
                f"razao {linha['razao_dtw_euclidiana']:.3f}"
            )

        print(f"  CSV salvo : {arq_csv}")
        print(f"  PNG salvo : {arq_png}")
    else:
        print("  Sao necessarios pelo menos dois pesos para comparar.")

    # ------------------------------------------------------------------
    # 4) Caminho de alinhamento do par mais diferente
    # ------------------------------------------------------------------
    print("\n[4] Caminho de alinhamento (par com maior DTW)")
    print("-" * 70)

    arq_alinhamento = PASTA / f"dtw-alinhamento-{sufixo}.png"
    plotar_alinhamento(
        grade,
        medias,
        linhas_entre_pesos,
        arq_alinhamento,
        f"Caminho de alinhamento DTW - {regiao}",
    )
    if linhas_entre_pesos:
        print(f"  PNG salvo : {arq_alinhamento}")

    # ------------------------------------------------------------------
    # 5) Resumo final
    # ------------------------------------------------------------------
    print("\n[5] Resumo final")
    print("-" * 70)

    if not resumo_repetibilidade.empty:
        mais_repetivel = resumo_repetibilidade.loc[
            resumo_repetibilidade["dtw_media"].idxmin()
        ]
        print(
            "  Peso mais repetivel: "
            f"{rotulo_peso(int(mais_repetivel['peso']))} "
            f"(DTW media {mais_repetivel['dtw_media']:.2f})"
        )

    if linhas_entre_pesos:
        par_mais_distante = max(
            linhas_entre_pesos, key=lambda linha: linha["distancia_dtw"]
        )
        print(
            "  Par de pesos mais diferente: "
            f"{rotulo_peso(par_mais_distante['peso_a'])} x "
            f"{rotulo_peso(par_mais_distante['peso_b'])} "
            f"(DTW {par_mais_distante['distancia_dtw']:.2f})"
        )

    print("\nConcluido.")


if __name__ == "__main__":
    main()
