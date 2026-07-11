"""Perfil estruturado de longo prazo (nome, preferencias, projetos, rotina,
habitos, objetivos) em SQLite.

SQLite puro (stdlib) em vez de um dependencia extra: e durável, facil de
inspecionar manualmente durante desenvolvimento (`sqlite3 data/profile.db`),
e as leituras/escritas aqui sao pequenas e pouco frequentes (uma leitura por
turno na construcao do prompt, escritas so na consolidacao em background).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class ProfileStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO profile_facts (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value),
        )
        self._conn.commit()

    def get_all(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM profile_facts ORDER BY key").fetchall()
        return {key: value for key, value in rows}

    def summary(self) -> str | None:
        facts = self.get_all()
        if not facts:
            return None
        return "\n".join(f"- {key}: {value}" for key, value in facts.items())

    def close(self) -> None:
        self._conn.close()
