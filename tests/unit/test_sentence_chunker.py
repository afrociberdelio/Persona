from persona.llm.sentence_chunker import SentenceChunker


def _feed_all(chunker: SentenceChunker, tokens: list[str]) -> list[str]:
    sentences = []
    for token in tokens:
        sentences.extend(chunker.feed(token))
    return sentences


def test_emits_sentence_on_boundary_with_trailing_space():
    chunker = SentenceChunker(min_chunk_chars=4)
    sentences = _feed_all(chunker, ["Ola, ", "tudo bem", "? ", "Aqui vai o resto"])

    assert sentences == ["Ola, tudo bem?"]
    assert chunker.flush() == "Aqui vai o resto"


def test_does_not_split_on_short_abbreviation_like_fragment():
    chunker = SentenceChunker(min_chunk_chars=8)
    sentences = _feed_all(chunker, ["Dr", ". ", "Silva chegou", ". "])

    # "Dr." sozinho e curto demais (< min_chunk_chars) entao nao e emitido
    # como sentenca isolada -- funde com o restante ate o proximo limite valido.
    assert sentences == ["Dr. Silva chegou."]


def test_flush_returns_none_when_buffer_empty():
    chunker = SentenceChunker()
    assert chunker.flush() is None


def test_flush_returns_remaining_text_without_terminal_punctuation():
    chunker = SentenceChunker()
    chunker.feed("Uma frase sem ponto final")
    assert chunker.flush() == "Uma frase sem ponto final"
    assert chunker.flush() is None


def test_multiple_sentences_in_one_feed():
    chunker = SentenceChunker(min_chunk_chars=4)
    sentences = chunker.feed("Oi! Tudo bem? Vamos comecar. ")

    assert sentences == ["Oi!", "Tudo bem?", "Vamos comecar."]
