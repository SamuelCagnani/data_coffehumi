#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot-series-temporais-media.py
==============================

Objetivo
--------
Gerar UM grafico de series temporais da corrente (A) em funcao do tempo
decorrido (s), plotando a MEDIA POR QUILOGRAMA dos ensaios.

Cada pasta guarda varios ensaios separados por peso (kg). Este script:

  1. Le todos os arquivos .csv que estao na MESMA pasta deste script;
  2. Converte a coluna "Tempo" (HH:MM:SS.mmm) em tempo decorrido (s),
     contado a partir da primeira amostra de cada ensaio. Isso alinha os
     ensaios no instante zero e permite compara-los mesmo com duracoes
     diferentes;
  3. Agrupa os ensaios por peso (extraido do nome do arquivo, ex.: "220kg");
  4. Interpola todos os ensaios de um mesmo peso numa grade de tempo comum;
  5. Calcula a media ponto a ponto entre os ensaios de cada peso;
  6. Plota uma linha por peso com a media (sem area sombreada).

Formato esperado dos CSVs (saida da etapa de pre-processamento):

    Tempo,Corrente
    15:55:47.209,0.0
    15:55:47.280,0.0
    ...

Como executar
-------------
    python plot-series-temporais-media.py

Dependencias: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
# Backend "Agg" = nao abre janela; apenas desenha e salva o arquivo de imagem.
# Isso torna o script reproducivel em qualquer maquina (inclusive sem tela).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Garante saida UTF-8 no terminal do Windows (evita erro ao imprimir acentos).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Configuracoes (tudo que e "numero magico" fica aqui, facil de ajustar)
# ---------------------------------------------------------------------------
PASTA = Path(__file__).resolve().parent   # pasta onde este script esta salvo
PADRAO_CSV = "*.csv"                      # quais arquivos ler
SEPARADOR = ","                           # separador de colunas do CSV
COLUNA_TEMPO = "Tempo"                    # nome da coluna de tempo
COLUNA_CORRENTE = "Corrente"              # nome da coluna de corrente
FORMATO_TEMPO = "%H:%M:%S.%f"             # formato: hora:min:seg.milissegundos
PASSO_GRADE_S = 0.05                      # passo da grade de tempo comum (s)
DPI = 150                                 # resolucao da imagem salva


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------
def sufixo_figura(pasta: Path) -> str:
    """Cria um sufixo curto para o nome da imagem a partir do nome da pasta.

    Ex.: 'medidas-molhadas' -> 'molhadas' ; 'medidas-secas' -> 'secas'.
    """
    nome = pasta.name.lower()
    if "molhada" in nome:
        return "molhadas"
    if "seca" in nome:
        return "secas"
    return re.sub(r"\W+", "-", pasta.name)


def nome_amigavel(pasta: Path) -> str:
    """Titulo legivel para o grafico, derivado do nome da pasta."""
    if "molhada" in pasta.name.lower():
        return "Pedra Molhada"
    if "seca" in pasta.name.lower():
        return "Pedra Seca"
    return pasta.name


def extrair_peso(nome_arquivo: str) -> int | None:
    """Extrai o peso (kg) a partir do nome do arquivo usando expressao regular.

    Exemplos:
        '220kgPedraMolhada.csv'    -> 220
        '220kgPedraMolhada(2).csv' -> 220
        '0kgPedraSeca(5).csv'      -> 0
        '100kgPedraSeca(3).csv'    -> 100

    Retorna None se o nome nao contiver o padrao "<numero>kg".
    """
    encontrado = re.search(r"(\d+)\s*kg", nome_arquivo, flags=re.IGNORECASE)
    return int(encontrado.group(1)) if encontrado else None


def carregar_ensaio(caminho: Path) -> pd.DataFrame:
    """Le um CSV e devolve um DataFrame com 'tempo_s' e a coluna de corrente.

    A coluna original 'Tempo' (HH:MM:SS.mmm) e convertida para tempo
    decorrido em segundos, tomando como zero a primeira amostra do ensaio.
    """
    df = pd.read_csv(caminho, sep=SEPARADOR)

    colunas_faltando = [
        c for c in (COLUNA_TEMPO, COLUNA_CORRENTE) if c not in df.columns
    ]
    if colunas_faltando:
        raise ValueError(
            f"Colunas {colunas_faltando} nao encontradas em '{caminho.name}'."
        )

    instantes = pd.to_datetime(df[COLUNA_TEMPO], format=FORMATO_TEMPO)
    df["tempo_s"] = (instantes - instantes.iloc[0]).dt.total_seconds()

    return df[["tempo_s", COLUNA_CORRENTE]].copy()


def interpolar_na_grade(ensaio: pd.DataFrame, grade: np.ndarray) -> np.ndarray:
    """Reprojeta um ensaio na grade de tempo comum por interpolacao linear.

    Fora do intervalo coberto pelo ensaio, o valor e NaN (vazio). Assim, ao
    empilhar varios ensaios, as regioes sem dado nao contaminam a media.
    """
    return np.interp(
        grade,
        ensaio["tempo_s"].to_numpy(),
        ensaio[COLUNA_CORRENTE].to_numpy(),
        left=np.nan,
        right=np.nan,
    )


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    arquivos = sorted(PASTA.glob(PADRAO_CSV))
    if not arquivos:
        raise SystemExit(f"Nenhum arquivo CSV encontrado em: {PASTA}")

    # --- 1) Agrupar os arquivos por peso (kg) ----------------------------
    grupos: dict[int, list[Path]] = {}
    for arquivo in arquivos:
        peso = extrair_peso(arquivo.name)
        if peso is None:
            print(f"[aviso] ignorando arquivo sem peso no nome: {arquivo.name}")
            continue
        grupos.setdefault(peso, []).append(arquivo)

    print(f"Pasta analisada : {PASTA}")
    print(f"Arquivos lidos  : {len(arquivos)}")
    print(f"Pesos encontrados: {sorted(grupos)}")
    print("-" * 60)

    # --- 2) Preparar a figura -------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    # Paleta de cores: uma cor por peso, bem distribuida no espectro 'viridis'.
    cores = plt.cm.viridis(np.linspace(0.0, 0.92, max(len(grupos), 2)))

    # --- 3) Para cada peso, montar a media (e faixa min-max) -------------
    for cor, peso in zip(cores, sorted(grupos)):
        ensaios = [carregar_ensaio(c) for c in grupos[peso]]

        # Grade comum do grupo: de 0 ate a maior duracao desse peso.
        maior_duracao = max(e["tempo_s"].iloc[-1] for e in ensaios)
        grade = np.arange(0.0, maior_duracao + PASSO_GRADE_S, PASSO_GRADE_S)

        # Matriz no formato [n_ensaios x n_amostras], com NaN onde nao ha dado.
        matriz = np.vstack([interpolar_na_grade(e, grade) for e in ensaios])

        # Descarta instantes em que NENHUM ensaio tem dado (evita "slice vazio").
        com_dado = ~np.all(np.isnan(matriz), axis=0)
        grade, matriz = grade[com_dado], matriz[:, com_dado]

        media = np.nanmean(matriz, axis=0)   # media ponto a ponto entre ensaios

        rotulo = f"{peso} kg (n={len(ensaios)})"
        ax.plot(grade, media, color=cor, linewidth=2.0, label=rotulo)

        duracoes = [e["tempo_s"].iloc[-1] for e in ensaios]
        print(
            f"  {peso:>3} kg | {len(ensaios)} ensaio(s) | "
            f"duracao media {np.mean(duracoes):5.1f} s | "
            f"corrente media {np.nanmean(media):4.2f} A"
        )

    # --- 4) Acabamento do grafico ---------------------------------------
    ax.set_title(
        f"Series temporais da corrente - {nome_amigavel(PASTA)} (media por kg)"
    )
    ax.set_xlabel("Tempo decorrido (s)")
    ax.set_ylabel("Corrente (A)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Peso", ncol=2, fontsize=9)
    fig.tight_layout()

    # --- 5) Salvar ------------------------------------------------------
    destino = PASTA / f"series-temporais-media-{sufixo_figura(PASTA)}.png"
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    print("-" * 60)
    print(f"Grafico salvo em : {destino}")


if __name__ == "__main__":
    main()
