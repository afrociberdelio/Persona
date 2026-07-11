# Sobe o llama-server (llama.cpp, build CUDA) com o Qwen3-8B-Instruct GGUF.
#
# Pre-requisito: llama.cpp compilado com suporte CUDA para Windows, com
# llama-server.exe em $LlamaServerExe (ajuste o caminho abaixo) ou no PATH.
#
# --parallel 2: um slot para o turno de conversa ao vivo, outro reservado
# para a consolidacao de memoria em background -- assim ela nunca fica na
# fila atras de uma resposta falada (ver persona/memory/consolidator.py).

param(
    [string]$LlamaServerExe = "llama-server.exe",
    [string]$ModelPath = "$PSScriptRoot\..\models\qwen3-8b-instruct-q4_k_m.gguf",
    [int]$Port = 8080,
    [int]$ContextSize = 6144,
    [int]$Parallel = 2,
    [int]$NGpuLayers = 999
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
    --flash-attn
