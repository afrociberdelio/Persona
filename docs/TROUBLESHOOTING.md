# Problemas conhecidos

Issues reais encontrados e corrigidos durante o desenvolvimento deste
projeto — deixados aqui porque podem reaparecer em setups diferentes (outra
versao de lib, outro SO, etc).

## `scripts/install_models.py` falha ao baixar o LLM ou o Kokoro

Os nomes de repositorio/arquivo no topo do script (`LLM_REPO_ID`,
`LLM_FILENAME`, `KOKORO_REPO_ID`, etc) sao um ponto de partida razoavel, nao
uma garantia — quantizacoes GGUF de terceiros no Hugging Face mudam de nome
com frequencia, e repositorios podem ser reorganizados. Se o download
falhar com 404:

1. Procure em huggingface.co pelo repositorio atual do Qwen3-8B-Instruct em
   GGUF (busque por "Qwen3-8B-Instruct GGUF")
2. Atualize `LLM_REPO_ID`/`LLM_FILENAME` no script
3. O mesmo vale para os assets do Kokoro (`KOKORO_REPO_ID`)

Os nomes de arquivo **finais** em `models/` (`kokoro-v1.0.onnx`,
`voices-v1.0.bin`) precisam bater com `config/default.yaml`
(`tts.model_path`/`tts.voices_path`) — se voce baixar manualmente, renomeie
para esses nomes ou ajuste a config.

## `fastembed` reclama que o modelo de embedding "nao e suportado"

```
ValueError: Model intfloat/multilingual-e5-small is not supported in TextEmbedding.
```

`fastembed` so suporta `intfloat/multilingual-e5-large` (mais pesado,
1024 dimensoes), nao a versao `-small`. O default do projeto ja usa
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384
dimensoes, leve, cobre PT-BR bem) — se voce mudou
`memory.embedder_model` em `config/local.yaml` para outro valor, confira a
lista de modelos suportados:

```powershell
uv run python -c "from fastembed import TextEmbedding; [print(m['model']) for m in TextEmbedding.list_supported_models()]"
```

## Ferramentas MCP retornam hora em UTC mesmo pedindo outro fuso

No Windows, o modulo `zoneinfo` da stdlib **nao tem o banco de dados IANA
embutido** — ele so encontra fusos horarios como `America/Sao_Paulo` se o
pacote `tzdata` estiver instalado (ja e uma dependencia do projeto). Se
algum outro script/ferramenta nova usar `zoneinfo` diretamente, garanta que
roda dentro do venv do projeto (`uv run ...`), nao com um Python do sistema
sem `tzdata`.

## VRAM não é suficiente para os três modelos simultaneamente

Se `nvidia-smi` mostrar estouro de VRAM (ou `llama-server`/faster-whisper
falharem com erro de alocacao CUDA), a ordem recomendada de ajuste — sem
sacrificar o full-duplex, que e o diferencial do projeto — via
`config/local.yaml`:

1. Reduza `llm.context_size` e/ou `llm.parallel_slots` (menos KV cache)
2. Troque `stt.model_size` para `medium` ou `small` (faster-whisper)
3. So entao considere um LLM menor (ex: Qwen3-4B-Instruct) — isso muda
   `llm.model_name` e exige rebaixar o GGUF correspondente

Evite mover STT/TTS para CPU como primeira alternativa — isso quebra a meta
de latencia baixa citada na proposta original do projeto (<300ms de STT) e
tem efeito cascata em toda latencia do turno.

## `sounddevice` não encontra o dispositivo certo

Dispositivos de audio no Windows tem nomes as vezes duplicados entre APIs
diferentes (MME, DirectSound, WASAPI, WDM-KS) — o mesmo microfone fisico
aparece varias vezes. Para listar todos e escolher pelo indice/nome exato:

```powershell
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

Prefira as entradas `Windows WASAPI` quando disponiveis (melhor
suporte a baixa latencia no `sounddevice`/PortAudio no Windows).

## `VIRTUAL_ENV` warning do `uv`

```
warning: `VIRTUAL_ENV=...` does not match the project environment path `.venv`
```

Inofensivo — normalmente aparece se voce tem uma variavel de ambiente
`VIRTUAL_ENV` de outro projeto/venv ainda setada no shell. `uv` ignora e
usa o `.venv` correto do projeto de qualquer forma. Para eliminar o aviso,
rode `Remove-Item Env:\VIRTUAL_ENV` no PowerShell antes de usar `uv run`,
ou simplesmente ignore.
