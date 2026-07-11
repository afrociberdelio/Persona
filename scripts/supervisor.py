"""Watchdog simples: relanca o processo do Persona se ele cair, com backoff.

Por que nao um Windows Service de verdade para o processo do Persona:
servicos rodam na Session 0, que historicamente nao tem acesso confiavel
aos dispositivos de audio do usuario interativo -- quebraria captura de
mic/playback. Por isso o modelo de deploy recomendado e Task Scheduler
("run only when user is logged on", disparado no logon) + este watchdog
para auto-restart, e nao um servico.

(O llama-server, por outro lado, nao precisa de audio -- ele pode
perfeitamente rodar como um Windows Service via NSSM se for conveniente.)

Uso:
    python scripts/supervisor.py -- uv run persona
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "logs"
LOG_FILE = LOG_DIR / "supervisor.log"

MAX_BACKOFF_S = 60
INITIAL_BACKOFF_S = 2


def _log(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} [supervisor] {message}"
    print(line)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_forever(command: list[str]) -> None:
    backoff = INITIAL_BACKOFF_S
    while True:
        _log(f"iniciando: {' '.join(command)}")
        started_at = time.monotonic()
        try:
            proc = subprocess.Popen(command)
            exit_code = proc.wait()
        except KeyboardInterrupt:
            _log("interrompido pelo usuario, encerrando")
            return
        except FileNotFoundError as exc:
            _log(f"comando nao encontrado: {exc}")
            return

        uptime_s = time.monotonic() - started_at
        _log(f"processo encerrou (codigo={exit_code}, uptime={uptime_s:.1f}s)")

        # Uptime razoavel antes de cair -> reseta o backoff; crash rapido em
        # sequencia -> aumenta o backoff, para nao martelar reinicios.
        if uptime_s > MAX_BACKOFF_S:
            backoff = INITIAL_BACKOFF_S
        else:
            backoff = min(backoff * 2, MAX_BACKOFF_S)

        _log(f"reiniciando em {backoff}s...")
        time.sleep(backoff)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    command = args or ["uv", "run", "persona"]
    run_forever(command)


if __name__ == "__main__":
    main()
