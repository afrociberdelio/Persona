# Arquitetura do Persona

## Visao geral

Tudo roda como tasks `asyncio` dentro de **um unico processo Python**, exceto
o LLM, que roda como processo separado (`llama-server.exe`, build CUDA) por ja
expor uma API HTTP/SSE propria. Essa escolha evita overhead de IPC no
caminho critico de latencia (audio → VAD → coordenador) mantendo o
componente mais pesado isolado num processo a parte.

- **Event bus**: implementacao propria e minima
  (`src/persona/bus/event_bus.py`, ~50 linhas) sobre `asyncio.Queue` —
  publish/subscribe por topico, sem broker externo. Kafka/RabbitMQ/ZeroMQ
  seriam overengineering nessa escala (processo unico, usuario unico); os
  contratos de evento (`src/persona/bus/events.py`) ficam desacoplados o
  suficiente para trocar de transporte depois, se um dia for necessario
  distribuir os servicos entre maquinas.
- Chamadas bloqueantes (`faster-whisper.transcribe()`, inferencia Kokoro)
  sempre rodam via `loop.run_in_executor(ThreadPoolExecutor, ...)` para
  nunca travar o loop `asyncio`.
- **Audio**: `sounddevice` (WASAPI) com streams orientados a callback; o
  callback do PortAudio so empilha/desempilha PCM cru — toda logica real
  fica numa task `asyncio` consumidora, para nunca causar xruns/dropouts.

## Servicos e responsabilidades

| Servico | Responsabilidade | Onde roda | Modulo |
|---|---|---|---|
| Audio I/O | Captura de mic, playback, buffering de baixo nivel | asyncio + thread de callback do sounddevice | `audio/capture.py`, `audio/playback.py` |
| VAD | Silero VAD (ONNX, **CPU**) + onset/offset com hangover | CPU — nunca compete com a GPU para detectar interrupcao | `audio/vad.py` |
| STT | Segmentacao guiada por VAD + faster-whisper | GPU, via executor | `stt/segmenter.py`, `stt/whisper_engine.py` |
| LLM | Prompt, geracao streaming, cancelamento, tool-calling | Processo separado (`llama-server.exe`), via HTTP/SSE | `llm/llama_client.py` |
| TTS | Sintese Kokoro por sentenca + cancelamento | GPU, via executor | `tts/kokoro_engine.py`, `tts/synth_queue.py` |
| Memoria | Curto prazo, perfil (SQLite), episodica (LanceDB), consolidacao | Majoritariamente CPU | `memory/` |
| Turn Coordinator | FSM central — unico componente que decide "o que acontece agora" | asyncio | `coordinator/state_machine.py`, `coordinator/orchestrator.py` |
| Tools (MCP) | Registro/dispatch de ferramentas | asyncio | `tools/registry.py`, `llm/mcp_client.py` |

## O mecanismo do barge-in: a maquina de estados de turno

`TurnStateMachine` (`coordinator/state_machine.py`) e **pura**: nao faz I/O,
nao conhece audio real nem GPU. Ela so decide, dado um evento e o estado
atual, qual e o proximo estado e **quais efeitos simbolicos** devem
acontecer. Isso e deliberado — permite testar cada transicao com um teste
unitario trivial (`tests/unit/test_state_machine.py`), sem mockar nada.

Estados: `IDLE → USER_SPEAKING → TRANSCRIBING → THINKING → SPEAKING`, com
`INTERRUPTED`/barge-in modelado como uma **transicao**, nao um estado de
repouso.

Invariante central: a captura de mic + VAD estao **sempre rodando**, em
qualquer estado. O mesmo evento `VADSpeechStart` tem significado diferente
dependendo so do estado atual:

- Em `IDLE`: comeca um turno novo.
- Em `USER_SPEAKING`/`TRANSCRIBING`: no-op (ja esta capturando).
- Em `THINKING` ou `SPEAKING`: e um **barge-in** — dispara, nessa ordem:
  1. `STOP_PLAYBACK` — limpa a fila de playback (quase instantaneo, pois o
     playback e orientado a fila/callback, nunca a escrita bloqueante — e
     isso que garante o alvo de <100ms independente da carga da GPU)
  2. `CANCEL_LLM` — cancela a task asyncio que consome o stream do LLM;
     isso lanca `CancelledError` dentro do `async with` do cliente HTTP
     (`llm/llama_client.py`), fechando a conexao — o `llama-server` detecta
     o disconnect e libera o slot de geracao imediatamente
  3. `CANCEL_TTS` — cancela jobs de sintese ainda nao concluidos na fila
     (`tts/synth_queue.py`); jobs ja em execucao terminam mas seu resultado
     e descartado (tag de `turn_id` obsoleto)
  4. `BEGIN_UTTERANCE` — comeca a capturar a fala nova

`orchestrator.py` e o unico lugar que traduz um `SideEffect` simbolico numa
acao real — e a camada de integracao entre a FSM pura e os servicos
concretos (audio, LLM, TTS, memoria).

Toda transicao e logada com timestamp monotonico
(`TurnStateMachine.history`), permitindo medir a latencia real de barge-in
em producao (`TurnStateMachine.last_barge_in_latency_s()`).

## Orcamento de VRAM (alvo: 12GB)

Os tres modelos (Whisper, Qwen3, Kokoro) ficam **residentes na VRAM durante
toda a execucao** — recarregar um LLM de 8B leva segundos, o que inviabiliza
a latencia de barge-in. O que e sequenciado e *computacao*, nao
*residencia*: STT computa durante `TRANSCRIBING`, LLM durante `THINKING`,
TTS durante a preparacao de `SPEAKING`, com sobreposicao breve no
pipelining LLM→TTS por sentenca.

Orcamento aproximado numa GPU de 12GB:

| Componente | VRAM aproximada |
|---|---|
| Qwen3-8B-Instruct, GGUF Q4_K_M (pesos + KV cache, contexto 6k) | ~6.5–7.5GB |
| Faster-Whisper `large-v3-turbo`, `int8_float16` | ~1.5GB |
| Kokoro (82M params) via ONNX + CUDAExecutionProvider | <0.5GB |
| Reserva WDDM/display do Windows | ~0.3–0.5GB |
| **Total** | **~10.5–11.5GB de 12GB** |

Justo mas viavel. Alavancas se nao couber, nesta ordem: reduzir
`llm.context_size`/`llm.parallel_slots` → trocar `stt.model_size` para
`medium`/`small` → so entao considerar um LLM menor (ver
`docs/TROUBLESHOOTING.md` para como aplicar isso via `config/local.yaml`
sem tocar no `default.yaml`, que reflete o alvo de producao).

Dois contextos CUDA, nao tres: STT+TTS ficam no mesmo processo Python
(`persona`), o LLM no seu proprio processo (`llama-server.exe`) — economiza
~0.5GB vs. rodar cada um em processo separado.

## Por que llama.cpp server, e nao vLLM/Ollama

- **vLLM**: suporte nativo a Windows e fraco (e Linux/WSL2-first) —
  desqualificado pelo alvo de rodar nativo no Windows.
- **Ollama**: e llama.cpp por baixo, mas seu comportamento padrao de
  descarregar modelos ociosos vai contra "ficar residente" (mitigavel com
  `keep_alive: -1`, mas e uma camada de abstracao a mais com menos controle
  direto de slots/contexto).
- **`llama-server.exe` puro**: endpoint OpenAI-compatible
  (`/v1/chat/completions`, SSE), controle direto de
  `--ctx-size`/`--parallel`/`--n-gpu-layers`, e o cancelamento necessario
  para o barge-in e simplesmente **fechar a conexao HTTP** — exatamente o
  que `llm/llama_client.py` faz quando a task e cancelada.

## Segmentacao de STT (fala longa)

`stt/segmenter.py` buferiza tudo desde o inicio da fala (VAD) ate o fim, e
transcreve o buffer inteiro de uma vez — cobre o caso comum. Para falas
longas (>20s configuravel), emite uma transcricao **parcial** (preview, ex:
para log) do trecho acumulado, mas a transcricao final e sempre feita
transcrevendo o buffer inteiro no fim da fala — nao por costura de texto
entre chunks sobrepostos. Uma politica de reconciliacao tipo LocalAgreement
adicionaria complexidade real (alinhamento de texto) que nao se paga no v1.

## Memoria: curto prazo + longo prazo

- **Curto prazo** (`memory/short_term.py`): janela deslizante das ultimas N
  trocas, mantida em memoria (nao persiste entre reinicios).
- **Longo prazo**, dois stores:
  - **Perfil estruturado** (`memory/profile_store.py`, SQLite): fatos
    estaveis — nome, preferencias, projetos, rotina, habitos, objetivos.
  - **Memoria episodica** (`memory/episodic_store.py`, LanceDB): resumos de
    conversas passadas, buscaveis por similaridade semantica
    (`memory/embedder.py`, FastEmbed multilingue, CPU).
- **Consolidacao** (`memory/consolidator.py`): disparada por um idle-timer
  no orchestrator (so depois de alguns segundos sem fala nova), chama o LLM
  para extrair fatos + um resumo episodico dos turnos recentes, e faz
  upsert nos dois stores. Roda num slot separado do `llama-server`
  (`llm.parallel_slots: 2` em `config/default.yaml`) para nunca competir
  com um turno de conversa ao vivo.

`llm/prompt_builder.py` monta a lista de mensagens por turno: system prompt
+ bloco de memoria (perfil + top-k trechos episodicos relevantes) + turnos
recentes de curto prazo + fala atual — com corte automatico (turnos mais
antigos primeiro, depois trechos episodicos) se estourar o orcamento de
tokens configurado (`llm.context_size`).

## MCP e ferramentas

`tools/registry.py` agrega multiplos servidores MCP (cada um um processo
filho falando o protocolo por stdio) e expoe seus schemas no formato
OpenAI-tools para o `llama_client`. `tools/demo_time_tool.py` e uma
ferramenta trivial (`get_current_time`) que prova o caminho ponta a ponta:
LLM pede a ferramenta → orchestrator despacha via MCP → resultado volta →
incorporado na resposta falada. Novas ferramentas (busca web, controle do
Windows, WhatsApp/Telegram) entram como novos servidores MCP registrados no
mesmo `ToolRegistry`, sem tocar em audio/VAD/coordenador.

## Extensoes futuras (fora do escopo da Fase 1)

A combinacao event-bus + MCP e o ponto de extensao intencional:

- **Visao computacional / compreensao de tela**: um novo servico `vision`
  publicando eventos de percepcao no mesmo bus.
- **Controle do Windows, WhatsApp/Telegram/e-mail, busca na web**: cada um
  um servidor MCP adicional registrado no `ToolRegistry` ja existente.
- **Agentes especializados**: um "cerebro" alternativo atras do servico de
  LLM, sem tocar em audio/VAD/coordenador.

## Status de verificacao

O que foi testado neste repositorio (sem GPU/audio real disponivel no
ambiente de desenvolvimento):

- 20 testes unitarios cobrindo toda a tabela de transicao da FSM (incluindo
  os dois caminhos de barge-in), o prompt builder (incluindo truncamento) e
  o sentence chunker (incluindo o caso de abreviacao/fragmento curto)
- Import de todos os modulos, incluindo os com dependencias pesadas (torch,
  onnxruntime, mcp) — confirma que a arvore de imports esta correta
- Scaffold MCP ponta a ponta real (subir servidor demo, listar ferramenta,
  chamar, receber resultado)
- VAD processando frames de audio sintetico (silencio/ruido)
- Pipeline de memoria ponta a ponta (embedder real + LanceDB + SQLite +
  consolidator com um LLM fake)

O que **nao** foi validado aqui e precisa de teste manual no servidor de
producao: latencia real de barge-in com audio de verdade, qualidade de voz
do Kokoro em PT-BR, orcamento de VRAM com os tres modelos concorrentes sob
carga real, e o fluxo de tool-calling com o Qwen3 de verdade (o teste MCP
aqui usou um fake). Ver checklist em `docs/DEPLOYMENT.md`.
