# Persona

Assistente de voz local, full-duplex, rodando inteiramente na sua propria maquina
(sem depender de nuvem). Combina:

- **STT**: Faster-Whisper
- **LLM**: Qwen3-8B-Instruct, servido via `llama-server` (llama.cpp)
- **TTS**: Kokoro (via `kokoro-onnx`)
- **VAD**: Silero VAD
- **Memoria em dois niveis**: curto prazo (janela deslizante) + longo prazo
  (perfil estruturado em SQLite + memoria episodica buscavel em LanceDB)
- **MCP** (Model Context Protocol) para integracao padronizada de ferramentas

O diferencial do projeto e o **full-duplex real**: enquanto a IA fala, o
microfone continua sendo monitorado; se voce comecar a falar, a resposta e
cortada em menos de 100ms e um turno novo comeca imediatamente — sem o
modelo tradicional de "espere sua vez" da maioria dos assistentes de voz.

Toda a arquitetura, as decisoes tecnicas e o porque de cada escolha estao
documentados em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Para colocar
isso no ar num servidor, veja [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).
Problemas conhecidos e como resolve-los: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Status atual

Fase 1 (loop de voz completo, full-duplex, memoria em dois niveis e scaffold
MCP) esta implementada. O que foi verificado neste repositorio (testes
unitarios da logica pura + smoke tests de cada modulo isoladamente) esta
descrito em [`docs/ARCHITECTURE.md#status-de-verificacao`](docs/ARCHITECTURE.md#status-de-verificacao).
O que so pode ser validado no hardware de producao (latencia real de
barge-in, qualidade de voz, orcamento de VRAM com os 3 modelos concorrentes)
ainda precisa de um teste manual fim-a-fim la — veja o checklist em
[`docs/DEPLOYMENT.md#checklist-de-validacao-pos-deploy`](docs/DEPLOYMENT.md#checklist-de-validação-pós-deploy).

## Requisitos

- Windows 10/11
- Python gerenciado via [`uv`](https://docs.astral.sh/uv/)
- GPU NVIDIA com CUDA — alvo de producao: **12GB VRAM** (ex: RTX 3060). Veja o
  orcamento de VRAM detalhado em `docs/ARCHITECTURE.md`
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) compilado com suporte
  CUDA (`llama-server.exe`)
- Um **headset** (microfone + fone de ouvido) — o design da Fase 1 assume
  ausencia de eco acustico entre caixas de som e microfone; caixas de som
  exigiriam cancelamento de eco (AEC), que nao esta implementado

## Instalacao

```powershell
git clone <url-do-seu-repositorio> Persona
cd Persona
uv sync
```

Baixe os modelos (LLM GGUF + assets do Kokoro):

```powershell
uv run python scripts/install_models.py
```

> Os nomes exatos de repositorio/arquivo de quantizacoes GGUF de terceiros
> mudam com frequencia — confira em huggingface.co e ajuste as constantes no
> topo de `scripts/install_models.py` se o download falhar (ver
> `docs/TROUBLESHOOTING.md`).

`faster-whisper` e `fastembed` baixam seus proprios modelos automaticamente
no primeiro uso (no primeiro turno de conversa e na primeira consolidacao de
memoria, respectivamente) — nao precisam de download manual.

## Configuracao

Toda a config vive em [`config/default.yaml`](config/default.yaml) (paths de
modelo, device de audio, thresholds de VAD, tamanho de contexto do LLM, etc).

Para diferencas especificas de maquina (por exemplo, um LLM menor num
notebook de desenvolvimento com GPU menor que a do servidor de producao),
crie `config/local.yaml` (gitignored) — os valores la sobrescrevem o
default, secao por secao. Exemplo:

```yaml
# config/local.yaml
llm:
  model_name: qwen3-4b-instruct-q4_k_m
stt:
  device: cpu
```

## Rodando

1. Suba o servidor do LLM primeiro:
   ```powershell
   .\scripts\run_llama_server.ps1
   ```
2. Em outro terminal, rode o Persona:
   ```powershell
   uv run persona
   ```

Para producao (auto-restart em caso de crash), veja
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — resumo rapido:

```powershell
python scripts/supervisor.py -- uv run persona
```

## Estrutura do projeto

```
src/persona/
  bus/          event bus minimo (asyncio.Queue) + eventos tipados
  coordinator/  maquina de estados de turno (mecanismo do barge-in) + orchestrator
  audio/        captura, playback (parada instantanea), VAD (Silero)
  stt/          faster-whisper + segmentacao guiada por VAD
  llm/          cliente llama-server, prompt builder, sentence chunker, cliente MCP
  tts/          kokoro-onnx + fila de sintese cancelavel
  memory/       curto prazo, perfil (SQLite), episodica (LanceDB), consolidacao
  tools/        registro de ferramentas MCP + ferramenta demo
config/         default.yaml (versionado) + local.yaml (gitignored, opcional)
scripts/        install_models.py, run_llama_server.ps1, supervisor.py
tests/unit/     testes puros (state machine, prompt builder, sentence chunker)
docs/           ARCHITECTURE.md, DEPLOYMENT.md, TROUBLESHOOTING.md
```

`models/` e `data/` sao gitignored (binarios grandes e estado especifico da
maquina) — veja `docs/DEPLOYMENT.md` para como popula-los no servidor.

## Testes

```powershell
uv run pytest tests/unit -v
```

Cobrem so a logica pura (maquina de estados, prompt builder, sentence
chunker) — nao dependem de GPU, audio ou modelos. Verificacao do
comportamento com audio/modelos reais e manual (ver checklist de deploy).
