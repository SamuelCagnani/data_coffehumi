#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot-diferencas-series.py
=========================

Objetivo
--------
Calcular e plotar as DIFERENCAS PONTO A PONTO entre as series temporais
medias de cada peso (kg), em um unico grafico.

A conta feita para cada par (pesoA, pesoB) e simplesmente:

    delta(t) = media_pesoA(t) - media_pesoB(t)

Como os ensaios tem duracoes e instantes de inicio diferentes, antes de
subtrair e preciso colocar as series numa MESMA base de tempo:

  1. Le todos os .csv da pasta onde este script esta salvo;
  2. Converte a coluna "Tempo" (HH:MM:SS.mmm) em tempo decorrido (s),
     contado da primeira amostra de cada ensaio;
  3. Agrupa por peso (extraido do nome, ex.: "220kg") e calcula a media
     ponto a ponto de cada peso numa grade de tempo comum;
  4. Limita tudo a uma JANELA COMUM = 0 ate a menor duracao entre TODOS os
     ensaios da pasta, garantindo que todos os arquivos contribuam em
     todos os instantes (subtracao valida em toda a extensao);
  5. Subtrai as medias dos pares definidos e plota (uma linha por par).

Pares de diferenca
------------------
- Pasta "seca"    : 100-vazio, 200-vazio, 200-100  (vazio = 0 kg)
- Pasta "molhada" : diferencas entre pesos CONSECUTIVOS
                   (ex.: 240-220, 260-240, ...)

Formato esperado dos CSVs (saida do pre-processamento):

    Tempo,Corrente
    15:55:47.209,0.0
    ...

Como executar
-------------
    python plot-diferencas-series.py

Dependencias: numpy, pandas, matplotlib.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
# Backend "Agg": nao abre janela; apenas salva a imagem (reprodutivel).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Saida UTF-8 no terminal do Windows (evita erro ao imprimir acentos).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
PASTA = Path(__file__).resolve().parent   # pasta onde este script esta salvo
PADRAO_CSV = "*.csv"
SEPARADOR = ","
COLUNA_TEMPO = "Tempo"
COLUNA_CORRENTE = "Corrente"
FORMATO_TEMPO = "%H:%M:%S.%f"             # HH:MM:SS.milissegundos
PASSO_GRADE_S = 0.05                      # passo da grade de tempo comum (s)
DPI = 150


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------
def nome_amigavel(pasta: Path) -> str:
    """Titulo legivel para o grafico, derivado do nome da pasta."""
    if "molhada" in pasta.name.lower():
        return "Pedra Molhada"
    if "seca" in pasta.name.lower():
        return "Pedra Seca"
    return pasta.name


def sufixo_figura(pasta: Path) -> str:
    """Sufixo curto do nome da imagem (ex.: 'medidas-secas' -> 'seca')."""
    nome = pasta.name.lower()
    if "molhada" in nome:
        return "molhada"
    if "seca" in nome:
        return "seca"
    return re.sub(r"\W+", "-", pasta.name)


def rotulo_peso(peso: int) -> str:
    """Rotulo amigavel do peso: 0 vira 'vazio'."""
    return "vazio" if peso == 0 else f"{peso} kg"


def extrair_peso(nome_arquivo: str) -> int | None:
    """Extrai o peso (kg) do nome do arquivo, ex.: '220kgPedraMolhada(2).csv'."""
    encontrado = re.search(r"(\d+)\s*kg", nome_arquivo, flags=re.IGNORECASE)
    return int(encontrado.group(1)) if encontrado else None


def carregar_ensaio(caminho: Path) -> pd.DataFrame:
    """Le um CSV e devolve 'tempo_s' (segundos decorridos) e 'Corrente'."""
    df = pd.read_csv(caminho, sep=SEPARADOR)

    faltando = [c for c in (COLUNA_TEMPO, COLUNA_CORRENTE) if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas {faltando} nao encontradas em '{caminho.name}'.")

    instantes = pd.to_datetime(df[COLUNA_TEMPO], format=FORMATO_TEMPO)
    df["tempo_s"] = (instantes - instantes.iloc[0]).dt.total_seconds()
    return df[["tempo_s", COLUNA_CORRENTE]].copy()


def interpolar_na_grade(ensaio: pd.DataFrame, grade: np.ndarray) -> np.ndarray:
    """Reprojeta um ensaio na grade comum (NaN fora do intervalo do ensaio)."""
    return np.interp(
        grade,
        ensaio["tempo_s"].to_numpy(),
        ensaio[COLUNA_CORRENTE].to_numpy(),
        left=np.nan,
        right=np.nan,
    )


def definir_pares(pasta: Path, pesos: list[int]) -> list[tuple[int, int]]:
    """Define os pares (pesoA, pesoB) -> calcula media(A) - media(B).

    - Pasta seca : pares fixos relativos ao vazio (0 kg).
    - Pasta molhada: diferencas entre pesos consecutivos.
    """
    if "seca" in pasta.name.lower():
        # (numerador, subtraendo) na ordem em que serao plotados.
        return [(100, 0), (200, 0), (200, 100)]
    if "molhada" in pasta.name.lower():
        return [(atual, anterior) for anterior, atual in zip(pesos[:-1], pesos[1:])]
    # Caso generico: consecutivos.
    return [(atual, anterior) for anterior, atual in zip(pesos[:-1], pesos[1:])]


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main() -> None:
    arquivos = sorted(PASTA.glob(PADRAO_CSV))
    if not arquivos:
        raise SystemExit(f"Nenhum arquivo CSV encontrado em: {PASTA}")

    # --- 1) Carrega os ensaios e agrupa por peso -------------------------
    grupos: dict[int, list[pd.DataFrame]] = {}
    for arquivo in arquivos:
        peso = extrair_peso(arquivo.name)
        if peso is None:
            print(f"[aviso] ignorando arquivo sem peso no nome: {arquivo.name}")
            continue
        grupos.setdefault(peso, []).append(carregar_ensaio(arquivo))

    pesos = sorted(grupos)

    # --- 2) Janela comum = menor duracao entre TODOS os ensaios ----------
    duracoes = [e["tempo_s"].iloc[-1] for lista in grupos.values() for e in lista]
    janela = min(duracoes)
    grade = np.arange(0.0, janela, PASSO_GRADE_S)

    print(f"Pasta analisada : {PASTA}")
    print(f"Arquivos lidos  : {len(arquivos)} | Pesos: {pesos}")
    print(f"Janela comum    : 0 a {janela:.1f} s ({grade.size} pontos)")
    print("-" * 60)

    # --- 3) Media de cada peso na grade comum ----------------------------
    media_por_peso: dict[int, np.ndarray] = {}
    for peso in pesos:
        matriz = np.vstack([interpolar_na_grade(e, grade) for e in grupos[peso]])
        media_por_peso[peso] = np.nanmean(matriz, axis=0)

    # --- 4) Define e calcula as diferencas ponto a ponto -----------------
    pares = definir_pares(PASTA, pesos)
    for a, b in pares:
        if a not in media_por_peso or b not in media_por_peso:
            raise SystemExit(f"Par ({a}, {b}) requer peso inexistente na pasta.")

    fig, ax = plt.subplots(figsize=(11, 6))
    cores = plt.cm.viridis(np.linspace(0.0, 0.9, max(len(pares), 2)))

    for cor, (a, b) in zip(cores, pares):
        diferenca = media_por_peso[a] - media_por_peso[b]
        rotulo = f"{rotulo_peso(a)} - {rotulo_peso(b)}"
        ax.plot(grade, diferenca, color=cor, linewidth=1.8, label=rotulo)
        print(
            f"  {rotulo:>16} | media {np.nanmean(diferenca):+5.2f} A | "
            f"faixa {np.nanmin(diferenca):+5.2f} a {np.nanmax(diferenca):+5.2f} A"
        )

    # --- 5) Acabamento e salvamento --------------------------------------
    ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_title(
        f"Diferenca ponto a ponto das medias - {nome_amigavel(PASTA)}"
    )
    ax.set_xlabel("Tempo decorrido (s)")
    ax.set_ylabel("Diferenca de corrente (A)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Diferenca", ncol=2, fontsize=9)
    fig.tight_layout()

    destino = PASTA / f"diferencas-series-{sufixo_figura(PASTA)}.png"
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)
    print("-" * 60)
    print(f"Grafico salvo em : {destino}")


if __name__ == "__main__":
    main()
