"""Diagnostico: o headset Bluetooth aguenta microfone (Hands-Free) e
playback (A2DP) ativos ao mesmo tempo, ou o Windows for a a troca de
perfil e a saida fica muda?

Abre o microfone Bluetooth primeiro (o que normalmente forca o Windows a
negociar o perfil Hands-Free) e, com ele ainda ativo, toca um tom de teste
no endpoint A2DP (estereo, o que normalmente e usado pra audio "normal").
Se voce ouvir o tom claramente do inicio ao fim, o headset aguenta os dois
perfis simultaneos e o Persona pode usar os devices atuais sem mudanca.
Se nao ouvir nada (ou ouvir cortado/distorcido), confirma o conflito de
perfil A2DP/HFP -- ver docs/TROUBLESHOOTING.md.

Uso:
    uv run python scripts/test_bluetooth_duplex.py --list        # lista devices e sai
    uv run python scripts/test_bluetooth_duplex.py --input 1 --output 4
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd


def list_devices() -> None:
    print(sd.query_devices())


def run_test(input_device: int, output_device: int, duration_s: float = 4.0) -> None:
    in_info = sd.query_devices(input_device)
    out_info = sd.query_devices(output_device)
    print(f"Entrada: [{input_device}] {in_info['name']}")
    print(f"Saida:   [{output_device}] {out_info['name']}")

    def mic_callback(indata, frames, time_info, status):
        pass  # so mantem o stream vivo pra forcar o profile switch, descarta o audio

    print("\nAbrindo microfone Bluetooth (isso costuma forcar o perfil Hands-Free)...")
    in_stream = sd.InputStream(
        device=input_device, channels=1, samplerate=16000, callback=mic_callback
    )
    in_stream.start()
    time.sleep(1.5)  # da tempo pro Windows terminar a troca de perfil

    print(f"Tocando tom de teste (440Hz, {duration_s:.0f}s) na saida, com o mic AINDA ativo...")
    print(">>> Preste atencao: o tom toca do inicio ao fim, sem cortes? <<<")
    out_sr = int(out_info["default_samplerate"]) or 44100
    t = np.linspace(0, duration_s, int(out_sr * duration_s), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sd.play(tone, samplerate=out_sr, device=output_device, blocking=True)

    in_stream.stop()
    in_stream.close()
    print("\nTeste concluido.")
    print("- Ouviu o tom completo e limpo?      -> headset aguenta os dois perfis, sem mudanca necessaria")
    print("- Nao ouviu nada, ou veio cortado?    -> conflito A2DP/HFP confirmado, ver docs/TROUBLESHOOTING.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="lista os devices de audio e sai")
    parser.add_argument("--input", type=int, help="indice do device de entrada (mic)")
    parser.add_argument("--output", type=int, help="indice do device de saida")
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()

    if args.list or args.input is None or args.output is None:
        list_devices()
        if not args.list:
            print("\nInforme --input e --output com os indices do seu headset (ver lista acima).")
        return

    run_test(args.input, args.output, args.duration)


if __name__ == "__main__":
    main()
