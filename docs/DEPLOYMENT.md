# Deploy no servidor

Este guia assume o alvo de producao descrito no projeto: Windows nativo (sem
WSL2/Docker), GPU NVIDIA com ~12GB de VRAM, headset conectado ao servidor.

## 1. Preparar a maquina

- Driver NVIDIA + CUDA instalados, `nvidia-smi` funcionando
- [`uv`](https://docs.astral.sh/uv/) instalado
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) compilado com suporte
  CUDA (`llama-server.exe`) — confirme com `llama-server.exe --version` que
  ele reconhece a GPU
- Headset conectado (verifique em `Configuracoes de Som do Windows` que
  microfone e fone estao corretos e definidos, ou anote o nome exato do
  dispositivo para configurar em `config/local.yaml`)

## 2. Trazer o codigo

```powershell
git clone <url-do-seu-repositorio> C:\Persona
cd C:\Persona
uv sync
```

`models/` e `data/` **nao vem pelo git** (estao no `.gitignore` de proposito
— sao binarios grandes e estado especifico da maquina). Depois do clone:

```powershell
uv run python scripts/install_models.py
```

Se voce ja tinha os modelos baixados noutra maquina (ex: um GGUF de
quantizacao especifica que voce validou antes), e mais simples copiar a
pasta `models/` inteira via rede/USB do que rebaixar.

`data/` (SQLite do perfil + LanceDB da memoria episodica + logs) comeca
vazio num servidor novo — o perfil/memoria de longo prazo vai se formando
com o uso. Se voce ja tinha uma pasta `data/` de outra instancia e quer
preservar o que o Persona ja "sabe" sobre voce, copie ela tambem.

## 3. Config especifica do servidor

Se a config do servidor diferir de `config/default.yaml` (ex: nome exato do
dispositivo de audio, um device index especifico, paths de modelo
diferentes), crie `config/local.yaml` (gitignored, nao vai pro repositorio):

```yaml
# config/local.yaml
audio:
  input_device: "Microphone Array (Realtek(R) Audio)"
  output_device: "Headphones (Realtek HD Audio 2nd output)"
```

Para descobrir os nomes exatos dos dispositivos disponiveis no servidor:

```powershell
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 4. Subir o LLM

Teste manualmente primeiro:

```powershell
.\scripts\run_llama_server.ps1
```

Confirme que ele sobe sem erro e reconhece a GPU (log deve mostrar as
camadas carregadas na GPU). Pare com Ctrl+C.

Como o `llama-server` **nao precisa de acesso a audio**, ele pode
tranquilamente rodar como um Windows Service de verdade (Session 0), ao
contrario do processo do Persona (ver secao 5). Se quiser isso persistente
e resiliente a reinicio do servidor, use [NSSM](https://nssm.cc/) para
registra-lo como servico apontando para
`scripts\run_llama_server.ps1`.

## 5. Subir o Persona

**Por que nao um Windows Service para o processo do Persona**: servicos
rodam na Session 0, que historicamente nao tem acesso confiavel aos
dispositivos de audio do usuario interativo — quebraria a captura de
mic/playback. Por isso o modelo recomendado e:

1. **Task Scheduler** com um gatilho "at log on" e a opcao "Run only when
   user is logged on" (nao "Run whether user is logged on or not") — assim
   o processo roda na sessao interativa, com acesso normal ao audio.
2. A acao do agendamento aponta para `scripts/supervisor.py`, que relanca o
   processo automaticamente se ele cair, com backoff progressivo:

   ```
   Programa: C:\Persona\.venv\Scripts\python.exe
   Argumentos: C:\Persona\scripts\supervisor.py -- uv run persona
   Iniciar em: C:\Persona
   ```

Teste manualmente antes de agendar:

```powershell
python scripts/supervisor.py -- uv run persona
```

Logs vao para `data/logs/persona.log` (rotativo, configurado em
`persona.utils.logging_setup`) e `data/logs/supervisor.log` (o watchdog
em si).

## Checklist de validação pós-deploy

Isso e o que so pode ser testado com hardware/audio reais — nao foi (nem
podia ser) verificado durante o desenvolvimento:

- [ ] `llama-server` sobe e responde em `http://127.0.0.1:8080/health`
- [ ] `uv run persona` sobe sem erro e os logs mostram
      `Orchestrator no ar. Estado inicial: IDLE`
- [ ] Falar uma frase curta produz uma resposta audivel (Passo 1: loop
      basico, sem interromper)
- [ ] Interromper a IA no meio de uma resposta falada corta o audio quase
      instantaneamente e a fala nova e transcrita como turno novo (Passo 2:
      barge-in) — confira nos logs o timestamp entre `VADSpeechStart` e o
      `STOP_PLAYBACK` correspondente para validar a meta de <100ms
- [ ] Mencionar um fato (ex: seu nome) numa sessao, reiniciar o processo, e
      confirmar que uma sessao nova consegue recuperar esse fato (Passo 3:
      memoria) — inspecione `data/profile.db` diretamente
      (`sqlite3 data/profile.db "select * from profile_facts;"`) se a
      recuperacao nao parecer funcionar
- [ ] Perguntar algo que exija a ferramenta demo (`get_current_time`)
      aciona corretamente o tool-calling do Qwen3 de verdade (o teste deste
      repositorio usou um LLM fake, nao o modelo real)
- [ ] Qualidade de voz do Kokoro em PT-BR e aceitavel (era um risco
      conhecido levantado na analise inicial do projeto — Kokoro tem
      suporte mais forte em ingles)
- [ ] Uso de VRAM com os tres modelos carregados simultaneamente fica
      dentro do orcamento esperado (`nvidia-smi` durante uma conversa)
- [ ] Deslogar/logar (ou reiniciar o servidor) confirma que o Task
      Scheduler sobe o Persona automaticamente
- [ ] Matar o processo a forca (`taskkill /F /IM python.exe` — cuidado se
      houver outros processos Python rodando) confirma que o
      `supervisor.py` reinicia dentro de alguns segundos

## Atualizando o deploy

```powershell
cd C:\Persona
git pull
uv sync   # atualiza dependencias se pyproject.toml/uv.lock mudaram
```

Reinicie o processo do Persona (o supervisor detecta a queda e relanca; ou
pare/inicie manualmente a task no Task Scheduler). O `llama-server`
normalmente nao precisa reiniciar a menos que o modelo em si tenha mudado.
