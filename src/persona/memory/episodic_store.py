"""Memoria episodica: resumos de conversas passadas, buscaveis por similaridade.

LanceDB embutido (baseado em arquivo) em vez de Chroma: nao sobe processo
de servidor em background, o que combina com o principio geral do projeto
de nao adicionar servicos/dependencias que a escala (single-user, single-
machine) nao justifica.
"""
from __future__ import annotations

import lancedb
import numpy as np

_TABLE_NAME = "episodes"


class EpisodicStore:
    def __init__(self, path: str, embed_dim: int) -> None:
        self._db = lancedb.connect(path)
        self._embed_dim = embed_dim

        if _TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(_TABLE_NAME)
        else:
            seed = [{"text": "", "vector": [0.0] * embed_dim, "ts": ""}]
            self._table = self._db.create_table(_TABLE_NAME, data=seed)
            self._table.delete("text = ''")

    def add(self, text: str, vector: np.ndarray, ts: str) -> None:
        self._table.add([{"text": text, "vector": vector.astype(float).tolist(), "ts": ts}])

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[str]:
        if self._table.count_rows() == 0:
            return []
        results = self._table.search(query_vector.astype(float).tolist()).limit(top_k).to_list()
        return [row["text"] for row in results]
