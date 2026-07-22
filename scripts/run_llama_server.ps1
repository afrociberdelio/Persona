# Sobe o llama-server (llama.cpp, build CUDA) com o Qwen3-8B GGUF (texto
# puro -- nao usar a variante Qwen3-VL, ver scripts/install_models.py).
#
# Pre-requisito: llama.cpp compilado com suporte CUDA para Windows, com
# llama-server.exe em $LlamaServerExe (ajuste o caminho abaixo) ou no PATH.
#
# --parallel 2: um slot para o turno de conversa ao vivo, outro reservado
# para a consolidacao de memoria em background -- assim ela nunca fica na
# fila atras de uma resposta falada (ver persona/memory/consolidator.py).

param(
    [string]$LlamaServerExe = "llama-server.exe",
    [string]$ModelPath = "$PSScriptRoot\..\models\Qwen3-8B-Q4_K_M.gguf",
    [int]$Port = 8080,
    # --ctx-size e o TOTAL dividido entre os slots (--parallel) -- com 2
    # slots, cada um recebe metade disso. Aumentado de 6144 pra 12288
    # (6144/slot) porque os schemas das ferramentas MCP (23 no total, entre
    # filesystem/puppeteer/shell) por si so custam ~2000+ tokens de prompt
    # por chamada -- com 3072/slot (valor antigo) isso estourava o contexto
    # (erro "exceeds the available context size") assim que memoria/system
    # prompt entravam na conta. O --cache-type-k/-v q8_0 abaixo compensa boa
    # parte da VRAM extra que isso consome (mesma ordem de grandeza da
    # config original antes de ferramentas existirem).
    [int]$ContextSize = 12288,
    [int]$Parallel = 2,
    [int]$NGpuLayers = 999,
    # Builds recentes do llama.cpp trocaram --flash-attn de flag booleana
    # para exigir um valor (on/off/auto). Se seu build for mais antigo e
    # reclamar de argumento inesperado, edite este script e volte para
    # `--flash-attn` sem valor (flag booleana).
    [string]$FlashAttn = "on",
    # Quantiza o KV cache (q8_0 = ~metade da VRAM do cache vs f16 padrao,
    # perda de qualidade praticamente imperceptivel; q4_0 economiza mais
    # mas degrada mais). SO funciona com --flash-attn ligado (ja e o
    # default acima). Motivo de mexer nisso: libera VRAM pro Kokoro/Whisper
    # dividirem a GPU com folga, e reduz um pouco a banda de memoria usada
    # por token gerado.
    [string]$CacheTypeK = "q8_0",
    [string]$CacheTypeV = "q8_0"
)

if (-not (Test-Path $ModelPath)) {
    Write-Error "Modelo nao encontrado em $ModelPath. Rode scripts/install_models.py primeiro."
    exit 1
}

& $LlamaServerExe `
    --model $ModelPath `
    --host 127.0.0.1 `
    --port $Port `
    --ctx-size $ContextSize `
    --parallel $Parallel `
    --n-gpu-layers $NGpuLayers `
    --flash-attn $FlashAttn `
    --cache-type-k $CacheTypeK `
    --cache-type-v $CacheTypeV
