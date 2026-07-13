"""Baixa os artefatos de modelo que nao sao geridos automaticamente pelas libs.

O que este script baixa:
- GGUF do Qwen3-8B (para o llama-server) -> models/
- Modelo ONNX + banco de vozes do Kokoro -> models/

O que ele NAO precisa baixar (as libs cuidam disso sozinhas, com cache em
~/.cache/huggingface, no primeiro uso):
- faster-whisper (baixa e converte o checkpoint automaticamente)
- fastembed (baixa o modelo de embeddings automaticamente)

IMPORTANTE: use o Qwen3-8B de TEXTO puro, nao o Qwen3-VL-8B (vision-
language). O VL e pensado pra rodar junto com um arquivo mmproj e
conversoes GGUF de terceiros pra uso so-texto costumam ter problemas de
tokenizer/chat template (sintoma tipico: o modelo "repete" o que voce
falou em vez de responder -- ver docs/TROUBLESHOOTING.md). Os valores
abaixo apontam pro repositorio oficial da Qwen e foram conferidos via API
do Hugging Face (arquivo confirmado presente no momento em que este
comentario foi escrito) -- ainda assim, releases mudam, confira de novo se
o download falhar.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Repositorio oficial da Qwen (texto puro, nao VL) -- Qwen3 nao usa mais o
# sufixo "-Instruct" separado, o checkpoint "Qwen3-8B" ja e o chat-tuned.
LLM_REPO_ID = "Qwen/Qwen3-8B-GGUF"
LLM_FILENAME = "Qwen3-8B-Q4_K_M.gguf"

# Assets do Kokoro. O modelo vem do onnx-community (export "onnx/model.onnx",
# precisao completa -- a variante quantizada "model_q4f16.onnx" pode nao
# existir nesse repo, confira antes de trocar). As vozes vem de um repo
# separado (rumbleFTW) que republica o "voices-v1.0.bin" da release oficial
# do projeto kokoro-onnx em formato compativel com a lib `kokoro-onnx` --
# o "voices.bin" do proprio onnx-community usa outro layout e nao carrega
# corretamente com `kokoro_onnx.Kokoro`.
#
# Nomes finais devem bater com config/default.yaml (tts.model_path/voices_path).
#
# NOTA: o modelo "onnx/model.onnx" usa a nomenclatura de input "input_ids"
# (export "novo"), o que aciona um bug conhecido da lib kokoro-onnx 0.5.0
# (envia o input "speed" como int32 quando o grafo espera float32). Isso e
# corrigido em tempo de execucao por um monkeypatch em
# `persona/tts/kokoro_engine.py` -- nao precisa trocar de modelo por causa
# disso. Ver docs/TROUBLESHOOTING.md.
KOKORO_MODEL_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_MODEL_FILENAME = "onnx/model.onnx"
KOKORO_VOICES_REPO_ID = "rumbleFTW/kokoro-v1.0-onnx"
KOKORO_VOICES_FILENAME = "voices-v1.0.bin"
KOKORO_MODEL_TARGET_NAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_TARGET_NAME = "voices-v1.0.bin"


def download_llm() -> None:
    print(f"Baixando LLM: {LLM_REPO_ID}/{LLM_FILENAME}")
    path = hf_hub_download(repo_id=LLM_REPO_ID, filename=LLM_FILENAME, local_dir=MODELS_DIR)
    print(f"  -> {path}")


def download_kokoro() -> None:
    print(f"Baixando modelo Kokoro de: {KOKORO_MODEL_REPO_ID}")
    downloaded_model = Path(
        hf_hub_download(repo_id=KOKORO_MODEL_REPO_ID, filename=KOKORO_MODEL_FILENAME, local_dir=MODELS_DIR)
    )
    target_model = MODELS_DIR / KOKORO_MODEL_TARGET_NAME
    if downloaded_model.resolve() != target_model.resolve():
        shutil.copy2(downloaded_model, target_model)
    print(f"  -> {target_model}")

    print(f"Baixando vozes Kokoro de: {KOKORO_VOICES_REPO_ID}")
    downloaded_voices = Path(
        hf_hub_download(repo_id=KOKORO_VOICES_REPO_ID, filename=KOKORO_VOICES_FILENAME, local_dir=MODELS_DIR)
    )
    target_voices = MODELS_DIR / KOKORO_VOICES_TARGET_NAME
    if downloaded_voices.resolve() != target_voices.resolve():
        shutil.copy2(downloaded_voices, target_voices)
    print(f"  -> {target_voices}")


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
