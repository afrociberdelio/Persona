"""Baixa os artefatos de modelo que nao sao geridos automaticamente pelas libs.

O que este script baixa:
- GGUF do Qwen3-8B-Instruct (para o llama-server) -> models/
- Modelo ONNX + banco de vozes do Kokoro -> models/

O que ele NAO precisa baixar (as libs cuidam disso sozinhas, com cache em
~/.cache/huggingface, no primeiro uso):
- faster-whisper (baixa e converte o checkpoint automaticamente)
- fastembed (baixa o modelo de embeddings automaticamente)

IMPORTANTE: os nomes exatos de repositorio/arquivo de quantizacoes GGUF de
terceiros mudam com frequencia. Confira em huggingface.co antes de rodar e
ajuste as constantes abaixo se necessario -- os valores aqui sao um ponto
de partida razoavel, nao uma garantia de que o arquivo ainda existe com
esse nome exato.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Ajuste conforme a quantizacao GGUF que voce escolher no Hugging Face.
LLM_REPO_ID = "Qwen/Qwen3-8B-Instruct-GGUF"
LLM_FILENAME = "qwen3-8b-instruct-q4_k_m.gguf"

# Assets do projeto kokoro-onnx (thewh1teagle/kokoro-onnx no GitHub/HF).
# Nomes finais devem bater com config/default.yaml (tts.model_path/voices_path).
KOKORO_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILENAME = "onnx/model_q4f16.onnx"
KOKORO_VOICES_FILENAME = "voices.bin"
KOKORO_MODEL_TARGET_NAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_TARGET_NAME = "voices-v1.0.bin"


def download_llm() -> None:
    print(f"Baixando LLM: {LLM_REPO_ID}/{LLM_FILENAME}")
    path = hf_hub_download(repo_id=LLM_REPO_ID, filename=LLM_FILENAME, local_dir=MODELS_DIR)
    print(f"  -> {path}")


def download_kokoro() -> None:
    print(f"Baixando Kokoro: {KOKORO_REPO_ID}")
    for filename, target_name in (
        (KOKORO_MODEL_FILENAME, KOKORO_MODEL_TARGET_NAME),
        (KOKORO_VOICES_FILENAME, KOKORO_VOICES_TARGET_NAME),
    ):
        downloaded_path = Path(hf_hub_download(repo_id=KOKORO_REPO_ID, filename=filename, local_dir=MODELS_DIR))
        target_path = MODELS_DIR / target_name
        if downloaded_path.resolve() != target_path.resolve():
            shutil.copy2(downloaded_path, target_path)
        print(f"  -> {target_path}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or ["llm", "kokoro"]

    if "llm" in targets:
        download_llm()
    if "kokoro" in targets:
        download_kokoro()

    print(
        "\nfaster-whisper e fastembed baixam seus proprios modelos "
        "automaticamente na primeira execucao do Persona -- nada a fazer aqui."
    )


if __name__ == "__main__":
    main()
