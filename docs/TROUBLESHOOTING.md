# Problemas conhecidos

Issues reais encontrados e corrigidos durante o desenvolvimento deste
projeto — deixados aqui porque podem reaparecer em setups diferentes (outra
versao de lib, outro SO, etc).

## `llama-server` retorna 400 `exceed_context_size_error` depois de adicionar ferramentas MCP

Log: `request (NNNN tokens) exceeds the available context size (3072 tokens)`.

Causa: `--ctx-size` no `run_llama_server.ps1` e o TOTAL dividido entre os
slots (`--parallel`), nao por slot. Com `--ctx-size 6144 --parallel 2`,
cada slot so tinha 3072 tokens. Os schemas JSON das ferramentas MCP contam
como tokens de prompt em toda chamada -- com as 23 ferramentas registradas
nesta sessao (filesystem + puppeteer + shell), so os schemas ja consomem
uma fatia grande desse orcamento (estimado 2000-3000+ tokens), sobrando
pouco pro system prompt + memoria + fala do usuario.

Corrigido subindo `--ctx-size` pra 12288 (6144/slot) e reduzindo
`llm.context_size` no `config/default.yaml` (o orcamento que o *nosso*
`PromptBuilder` usa pra decidir quantos turnos de curto prazo/trechos
episodicos incluir) pra 3000, deixando espaco pro que o llama-server
realmente precisa alem das nossas mensagens (schemas de ferramentas +
`max_tokens_per_turn` de geracao).

Se isso ainda estourar conforme a conversa cresce (mais turnos de memoria
de curto prazo acumulando), os proximos ajustes, nesta ordem: (1) subir
`--ctx-size` mais um pouco (atencao ao uso de VRAM via `nvidia-smi` -- o
KV cache quantizado em q8_0 ajuda mas nao e de graca), (2) reduzir
`llm.context_size` ainda mais, (3) expor menos ferramentas de uma vez --
nem todas as 14 do filesystem sao essenciais pro dia a dia (`read_media_file`,
`list_directory_with_sizes`, `directory_tree`, `list_allowed_directories`,
`read_multiple_files` sao as candidatas mais faceis de cortar se precisar).

## Ferramentas MCP de arquivos/web (`filesystem`, `puppeteer`) não conectam

Esses dois servidores rodam via `npx` (Node.js), não Python — pré-requisito
que não existia nas fases anteriores do projeto. Confirme:

```powershell
npx --version
```

Se não estiver instalado, baixe o Node.js LTS em nodejs.org. `main.py` já
trata falha de conexão de um servidor MCP individualmente (loga e segue
sem aquela ferramenta, não derruba o Persona inteiro) — se só uma das
ferramentas não aparecer, confira o log de startup pela mensagem "Falha ao
conectar servidor MCP".

Nomes de pacote npm no ecossistema MCP mudam de organização/nome de vez em
quando — se `@modelcontextprotocol/server-filesystem` ou
`@modelcontextprotocol/server-puppeteer` derem 404, procure o nome atual no
registro npm ou no repositório de servidores MCP de referência.

## Risco de segurança: terminal + navegação web juntos

O servidor `shell` (`persona/tools/shell_tool.py`) exige confirmação em
duas etapas (`propose_shell_command` → `confirm_shell_command`) antes de
executar qualquer comando — a confirmação só pode vir de uma fala *nova*
do usuário (turno seguinte), nunca do mesmo turno em que o comando foi
proposto, porque ferramentas MCP só são despachadas dentro de
`_run_llm_turn`, que só roda depois de um `STTFinal` real vindo do
microfone.

Isso reduz mas não elimina o risco de **prompt injection indireto**: como o
`puppeteer` traz conteúdo de páginas web pro contexto do LLM, uma página
maliciosa poderia conter texto tentando convencer o modelo a propor um
comando perigoso — a confirmação por voz ainda depende de você prestar
atenção ao que o assistente está pedindo pra confirmar, não é uma trava
automática. Se algo pedir confirmação pra um comando que você não esperava,
não confirme.

## Known Folders do Windows resolvidos errado (ferramenta de arquivos sem acesso a Documentos/Desktop/Downloads)

`persona/main.py::_known_folder_path` usa a API `SHGetKnownFolderPath` do
Windows (via `ctypes`) em vez de supor `Path.home() / "Documents"` etc. —
isso é necessario porque Windows localizado (PT-BR usa "Documentos"/"Área
de Trabalho") e redirecionamento de pastas pelo OneDrive fazem esses
caminhos literais não existirem de verdade em muitas instalações. Se o log
de startup mostrar "Pasta 'X' não encontrada", confira no Explorador de
Arquivos se essa pasta tem um local customizado (clique direito → Propriedades →
aba Local) e ajuste manualmente se for um caso atípico não coberto pela API.

## Nenhum audio sai, mesmo com STT/LLM/TTS funcionando perfeitamente nos logs (causa raiz principal)

Esse foi o bug mais dificil de achar do projeto -- o pipeline inteiro
funcionava (transcricao certa, resposta do LLM coerente, sintese do Kokoro
rapida e sem erro), mas nenhum audio saia, sem excecao nenhuma em lugar
nenhum.

Causa: `persona/coordinator/orchestrator.py` chama
`playback.mark_no_more_chunks(turn_id)` assim que o LLM termina de
**gerar o texto** -- o que costuma acontecer bem antes da sintese de TTS de
cada sentenca (assincrona, em paralelo, uma por vez) realmente terminar.
Em `AudioPlayback._callback` (`persona/audio/playback.py`), a condicao de
"turno terminou naturalmente" so checava `buffer vazio + no_more_chunks`,
sem saber se algum audio JA tinha sido enfileirado alguma vez. Resultado:
no primeiro callback de audio depois de `mark_no_more_chunks()`, com o
buffer ainda vazio (nenhuma sentenca tinha terminado de sintetizar ainda),
a condicao disparava na hora -- resetando `_current_turn_id` pra `None`
*antes* de qualquer audio real chegar. Todo `enqueue()` das sentencas que
terminavam de sintetizar depois disso era descartado silenciosamente
(o `turn_id` nao batia mais com o turno atual), sem log visivel (era
`logger.debug`, nao `INFO`).

Sintoma no log (se voce tiver o logging de FSM ativo): a linha
`FSM: THINKING + playback_naturally_stopped -> THINKING` aparece **antes**
de qualquer `Kokoro: sintese concluida` para aquele turno -- esse e o
tell-tale sign de que o turno "fechou" antes de qualquer audio existir.

Corrigido rastreando explicitamente `_any_chunk_enqueued` em
`AudioPlayback`: o "terminou naturalmente" so pode disparar depois que
pelo menos um chunk de audio de verdade foi enfileirado E o buffer esvaziou
de fato. Regressao coberta em `tests/unit/test_audio_playback.py` (4 dos 5
testes falham se essa condicao voltar a ignorar `_any_chunk_enqueued`).

Bug irmao, na mesma familia: `orchestrator.py` tambem aplicava a transicao
de FSM `LLMDone -> IDLE` incondicionalmente pelo mesmo motivo (texto
terminando antes do audio) -- uma vez em `IDLE`, todo `TTSAudioChunkReady`
subsequente cai num estado sem transicao mapeada e e descartado (no-op por
design da maquina de estados). Corrigido: essa transicao so e aplicada
quando a resposta gerou zero sentencas (caso realmente vazio); quando ha
sentencas, a FSM fica em `THINKING` ate o primeiro audio real via
`TTSAudioChunkReady` (`THINKING -> SPEAKING`). Regressao em
`tests/unit/test_orchestrator_tts_timing.py`.

Licao geral: qualquer sinal de "terminou"/"nao tem mais nada vindo" que
depende do LLM terminar de **gerar texto** precisa ser tratado como
independente de "toda a sintese de audio associada a esse texto tambem
terminou" -- sao dois processos assincronos concorrentes, e o texto quase
sempre termina primeiro.

## O modelo "repete" frases que voce falou em vez de responder

Sintoma: a conversa flui (STT transcreve certo, TTS fala normalmente), mas
o conteudo da resposta parece ecoar/repetir o que voce acabou de dizer em
vez de responder de verdade. Nao e bug do Persona nem, necessariamente, do
LLM em si — o suspeito mais provavel e o **arquivo GGUF errado**.

Use sempre o **Qwen3-8B de texto puro** (`Qwen/Qwen3-8B-GGUF` no Hugging
Face), nao o **Qwen3-VL-8B** (vision-language). O VL e feito pra rodar
junto com um arquivo `mmproj` e conversoes GGUF de terceiros pra uso
so-texto costumam ter tokenizer/chat template mal configurados. Um sinal
concreto disso no log de inicializacao do `llama-server`:

```
control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model
```

Esse aviso indica que o token de fim-de-turno nao esta corretamente
marcado no GGUF — se o modelo nao reconhece com confianca onde uma resposta
deveria terminar/comecar, ele tende a so continuar/ecoar o texto de entrada
em vez de responder de fato.

Correcao: baixe o Qwen3-8B correto —
`scripts/install_models.py` ja aponta pro repositorio certo
(`Qwen/Qwen3-8B-GGUF`, arquivo `Qwen3-8B-Q4_K_M.gguf`, confirmado via API do
Hugging Face). Rode `uv run python scripts/install_models.py llm` de novo e
aponte `run_llama_server.ps1` (ou a flag `-ModelPath`) pro arquivo novo.

Se quiser confirmar rapidamente ANTES de trocar o modelo, teste o LLM
isolado (sem STT/TTS no meio):

```powershell
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Diga oi e me diga seu nome.\"}],\"max_tokens\":50}'
```

Se a resposta já vier estranha/ecoando nesse teste direto, confirma que o
problema é do modelo/GGUF, não do código do Persona.

## `scripts/install_models.py` falha ao baixar o LLM ou o Kokoro

Os nomes de repositorio/arquivo no topo do script (`LLM_REPO_ID`,
`LLM_FILENAME`, `KOKORO_REPO_ID`, etc) sao um ponto de partida razoavel, nao
uma garantia — quantizacoes GGUF de terceiros no Hugging Face mudam de nome
com frequencia, e repositorios podem ser reorganizados. Se o download
falhar com 404:

1. Procure em huggingface.co pelo repositorio atual do Qwen3-8B-Instruct em
   GGUF (busque por "Qwen3-8B-Instruct GGUF")
2. Atualize `LLM_REPO_ID`/`LLM_FILENAME` no script
3. O mesmo vale para os assets do Kokoro (`KOKORO_MODEL_REPO_ID`/
   `KOKORO_VOICES_REPO_ID`) — o modelo e as vozes vem de repositorios
   diferentes de proposito (ver comentario no script)

Os nomes de arquivo **finais** em `models/` (`kokoro-v1.0.onnx`,
`voices-v1.0.bin`) precisam bater com `config/default.yaml`
(`tts.model_path`/`tts.voices_path`) — se voce baixar manualmente, renomeie
para esses nomes ou ajuste a config.

## Kokoro falha com `TypeError: __init__() got an unexpected keyword argument 'providers'`

A classe `Kokoro` da lib `kokoro-onnx` (0.5.0, a mais recente no PyPI) **nao
aceita** um parametro `providers` no construtor — ela escolhe o provider
sozinha, internamente, checando se o pacote `onnxruntime-gpu` esta instalado
ou lendo a variavel de ambiente `ONNX_PROVIDER`. `persona/tts/kokoro_engine.py`
ja trata isso corretamente (nao passa `providers=`, seta `ONNX_PROVIDER` em
vez disso) — se voce ver esse erro, provavelmente esta numa versao mais
antiga do arquivo; atualize para a versao atual do repositorio.

Para o Kokoro realmente rodar na GPU (nao so tentar), o pacote
`onnxruntime-gpu` precisa estar instalado no lugar do `onnxruntime` puro
(que nao tem `CUDAExecutionProvider` compilado). Os dois pacotes ocupam o
mesmo namespace de import (`onnxruntime`) e podem conflitar se instalados
juntos — o projeto depende de `onnxruntime` puro por padrao (usado tambem
por `fastembed` e `silero-vad`), entao pra habilitar GPU no Kokoro
especificamente:

```powershell
uv pip install onnxruntime-gpu --force-reinstall
uv run python -c "import onnxruntime as rt; print(rt.get_available_providers())"
```

Confirme que `CUDAExecutionProvider` aparece na lista. Se o Persona nao
conseguir usar CUDA (driver/CUDA/cuDNN incompativeis com o build do
`onnxruntime-gpu`), o Kokoro cai silenciosamente para
`CPUExecutionProvider` com um aviso no log — mais lento, mas nao quebra a
funcionalidade.

## Kokoro falha com `onnxruntime...InvalidArgument: Unexpected input data type. Actual: (tensor(int32)), expected: (tensor(float))`

Bug real da lib `kokoro-onnx` 0.5.0 (a mais recente no PyPI — nao ha versao
mais nova que corrija isso). Em `_create_audio`, quando o modelo ONNX usado
e um export "novo" (input chamado `input_ids`, caso do modelo baixado de
`onnx-community/Kokoro-82M-v1.0-ONNX`), a lib monta o input `speed` como
`int32`:

```python
"speed": np.array([speed], dtype=np.int32),   # bug: deveria ser float32
```

...mas esses exports esperam `float32` nesse input, e o ONNX Runtime rejeita
a chamada. `persona/tts/kokoro_engine.py` corrige isso com um monkeypatch
minimo (`_patch_session_dtypes`) que envolve `session.run` e forca os dtypes
corretos logo antes da sessao rodar de fato — funciona independente de qual
variante do modelo (`input_ids` ou `tokens`) foi baixada, e nao depende de
trocar de fonte do modelo. Se voce ver esse erro mesmo assim, confirme que
esta rodando a versao atual de `kokoro_engine.py` (o monkeypatch precisa
estar presente).

## Nenhum audio sai, e/ou o log mostra `Exception ignored from cffi callback` com `could not broadcast input array from shape (N,) into shape (1,)`

Sintoma traicoeiro: o pipeline inteiro roda sem erro visivel (STT
transcreve, LLM responde), mas nenhum audio e ouvido — e so depois de
varios turnos aparece esse traceback vindo de dentro do callback do
`sounddevice`. Ele nao "crasha" o processo porque exceçoes dentro do
callback nativo do PortAudio sao capturadas e apenas logadas
(`cffi callback ... ignored`) — o que significa que esse erro provavelmente
estava acontecendo **silenciosamente em todo chunk de audio**, desde o
primeiro turno, e por isso nunca saiu som nenhum.

Causa: o modelo ONNX do Kokoro, dependendo do export, retorna o audio com
uma dimensao de lote (`shape (1, N)`) em vez de plano (`shape (N,)`).
`len()` num array `(1, N)` retorna `1` (o tamanho do lote), nao `N` (as
amostras de audio de verdade) — isso corrompe o calculo de quantas amostras
copiar no preenchimento do buffer de playback
(`persona/audio/playback.py::_callback`).

Corrigido em dois pontos (achatando para 1D com `.reshape(-1)`):
`persona/tts/kokoro_engine.py::_synthesize_sync` (na fonte) e
`persona/audio/playback.py::AudioPlayback.enqueue` (defensivamente, na
fronteira de entrada do playback, para proteger contra qualquer fonte de
audio futura que tenha o mesmo problema). Se voce ver esse erro mesmo assim,
confirme que esta rodando a versao atual desses dois arquivos.

## Kokoro falha com `ValueError: all the input array dimensions except for the concatenation axis must match exactly` dentro de `kokoro_onnx/__init__.py`, em `np.concatenate(audio)`

Mesma causa raiz do item acima (saida de audio do ONNX com dimensao de lote
extra, `shape (1, N)`), mas se manifestando num lugar mais dificil de
corrigir: quando uma sentenca e longa o suficiente para a lib dividir em
varios lotes de fonemas (`Kokoro._split_phonemes`), cada lote e sintetizado
separadamente e a propria `Kokoro.create()` tenta
`np.concatenate(audio_parts)` esperando arrays 1D. Como os lotes tem
tamanhos (`N`) diferentes, arrays `(1, N_i)` nao concatenam (dimensao 1 nao
bate entre eles) e a lib quebra **antes** de devolver qualquer coisa pro
nosso codigo — um `.reshape(-1)` so no retorno de `_synthesize_sync` chega
tarde demais.

Corrigido junto com o item acima, no mesmo monkeypatch
(`persona/tts/kokoro_engine.py::_patch_session_dtypes`, que apesar do nome
tambem achata a saida): a correcao acontece logo apos CADA chamada
individual ao ONNX Runtime, antes da lib tentar concatenar os lotes.

## Resposta vem pela metade, confusa ou picotada (mesmo com o Qwen3-8B correto)

Duas causas prováveis, tratadas juntas:

1. **Modo de raciocínio do Qwen3**: o Qwen3 suporta um modo de "pensamento"
   estendido, onde o modelo gera um bloco `<think>...</think>` de
   raciocínio interno antes da resposta de verdade. Isso é ótimo pra
   tarefas de lógica/código, péssimo pra um assistente de voz: (a) você não
   quer ouvir o raciocínio interno falado em voz alta, e (b) o bloco de
   pensamento pode consumir a maior parte de `max_tokens`, deixando pouco
   ou nada pra resposta real — o que se parece exatamente com "resposta
   pela metade" ou "confusa". Corrigido em `persona/llm/llama_client.py`:
   `stream_chat`/`chat_once` agora enviam `chat_template_kwargs:
   {"enable_thinking": false}` por padrão, que é como o chat template do
   Qwen3 desliga esse modo. Requer um `llama-server` recente o suficiente
   pra repassar `chat_template_kwargs` — em versões antigas isso é
   ignorado silenciosamente (sem quebrar nada, só sem efeito).
2. **Kokoro rodando em CPU** (comum se `onnxruntime-gpu` não estiver
   instalado — ver seção sobre `CUDAExecutionProvider` mais acima): a
   síntese de cada sentença pode demorar o suficiente pra abrir um buraco
   de silêncio audível entre uma sentença e a próxima no playback, dando
   uma sensação de áudio "picotado". Isso não é um bug de lógica — é
   limitação de velocidade — mas o log agora ajuda a confirmar: toda
   sentença enviada pro TTS e todo áudio pronto pra tocar são logados em
   `persona/coordinator/orchestrator.py` (`sentenca N -> TTS` e `audio
   pronto -> playback`), então dá pra medir o intervalo real entre eles.

Pra diagnosticar de verdade (em vez de só suspeitar), o log agora imprime
a resposta completa do LLM por turno (`Turno ... resposta completa (N
chars): '...'`) — se aparecer texto de raciocínio/`<think>` ali, ou se a
resposta terminar cortada no meio de uma frase, dá pra confirmar a causa
exata em vez de adivinhar.

## `llama-server` reclama de argumento inesperado em `--flash-attn`

Builds mais recentes do llama.cpp trocaram `--flash-attn` de flag booleana
(`--flash-attn` sozinho, liga) para exigir um valor explicito
(`--flash-attn on|off|auto`). `scripts/run_llama_server.ps1` ja usa
`--flash-attn on` por padrao (parametro `-FlashAttn`, ajustavel). Se seu
build for mais antigo e reclamar do valor `on` como argumento inesperado,
edite o script e volte para a flag sem valor.

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
