"""Acumula tokens do LLM em streaming e libera sentencas completas.

Isso e o que permite o pipelining LLM->TTS (o TTS comeca a sintetizar a
primeira sentenca antes do LLM terminar de gerar a resposta inteira),
essencial para o time-to-first-audio ficar baixo.

Deliberadamente simples: so pontuacao de fim de frase + uma heuristica pra
nao cortar abreviacoes (“Dr.”, “Sr.”, numeros como "3.14") ao meio. A
heuristica so se aplica a ponto final ("."): "!" e "?" nunca sao usados em
abreviacoes, entao sempre fecham a sentenca, mesmo curta (“Oi!”, “Sim?”).
Para ".", uma sentenca curta demais e tratada como possivel abreviacao e a
busca continua adiante no buffer pelo proximo limite, em vez de parar --
sem isso, o mesmo ponto curto seria encontrado de novo a cada chamada e a
sentenca real nunca seria emitida.
"""
from __future__ import annotations

_SENTENCE_ENDERS = (".", "!", "?")


class SentenceChunker:
    def __init__(self, min_chunk_chars: int = 8) -> None:
        self.min_chunk_chars = min_chunk_chars
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Alimenta um token/fragmento novo; retorna 0+ sentencas prontas."""
        self._buffer += token
        ready: list[str] = []
        search_from = 0

        while True:
            boundary = self._find_boundary(self._buffer, search_from)
            if boundary is None:
                break

            terminator = self._buffer[boundary]
            sentence = self._buffer[: boundary + 1].strip()

            looks_like_abbreviation = terminator == "." and len(sentence) < self.min_chunk_chars
            if looks_like_abbreviation:
                # Nao consome nada do buffer -- so continua procurando o
                # PROXIMO limite mais adiante, sem descartar o texto.
                search_from = boundary + 1
                continue

            self._buffer = self._buffer[boundary + 1 :]
            ready.append(sentence)
            search_from = 0

        return ready

    def flush(self) -> str | None:
        """Chamar ao fim da geracao para pegar qualquer texto residual sem pontuacao final."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining or None

    @staticmethod
    def _find_boundary(text: str, start: int = 0) -> int | None:
        for i in range(start, len(text)):
            if text[i] in _SENTENCE_ENDERS:
                nxt = text[i + 1 : i + 2]
                if nxt == "" or nxt.isspace():
                    return i
        return None
